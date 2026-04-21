import json
import os
from functools import partial
from pathlib import Path

from flask import Flask, request, jsonify
import torch
from app.src.pipelines import LookupPipeline, FuzzyMatchPipeline, BM25OkapiPipeline, BiencoderPipeline
from app.src.format import PassthroughFormatter

app = Flask(__name__)


device = 'cpu' if not torch.cuda.is_available() else 'cuda'
method2pipeline = {
    'lookup': LookupPipeline,
    'levenshtein': partial(FuzzyMatchPipeline, method='levenshtein'),
    'jaro-winkler': partial(FuzzyMatchPipeline, method='jaro_winkler'),
    'token-sort-ratio': partial(FuzzyMatchPipeline, method='token_sort_ratio'),
    'token-set-ratio': partial(FuzzyMatchPipeline, method='token_set_ratio'),
    'bm25': BM25OkapiPipeline,
    'biencoder': partial(BiencoderPipeline, ner_version=2, device=device),
}

cdm2formatter = {
    'none': PassthroughFormatter
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_pipeline_params(data: dict):
    """Validate and extract lang/method/entities/negation from request dict.
    Returns (params_dict, None) on success or (None, error_str) on failure.
    """
    missing = [f for f in ('lang', 'method', 'entities') if f not in data]
    if missing:
        return None, f"Missing required field(s): {', '.join(repr(f) for f in missing)}"

    method = data['method']
    if method not in method2pipeline:
        return None, f"Unknown method '{method}'. Valid: {list(method2pipeline)}"

    return {
        'method': method,
        'lang': data['lang'],
        'entities': data['entities'],
        'negation': data.get('negation', False),
    }, None


def _build_pipeline(method, lang, entities, negation):
    pipeline_cls = method2pipeline[method]
    if method == 'biencoder':
        return pipeline_cls(lang=lang, entities=entities, negation=negation)
    return pipeline_cls(lang=lang, entities=entities)


def _normalize_texts(raw_texts: list):
    """Accept list of str or list of {text, metadata?} objects (mixed allowed).
    Returns (texts, metadatas, None) or (None, None, error_str).
    """
    texts, metadatas = [], []
    for item in raw_texts:
        if isinstance(item, str):
            texts.append(item)
            metadatas.append(None)
        elif isinstance(item, dict) and 'text' in item:
            texts.append(item['text'])
            metadatas.append(item.get('metadata'))
        else:
            return None, None, 'Each item in "texts" must be a string or {"text": ...} object'
    return texts, metadatas, None


def _run_pipeline(pipeline, texts: list, metadatas: list) -> list:
    formatter = cdm2formatter['none']()
    annotations = pipeline.predict(texts=texts)
    return [
        formatter.serialize(text, ann, meta)
        for text, ann, meta in zip(texts, annotations, metadatas)
    ]


def _write_to_dir(results: list, output_dir: str, filenames: list) -> list:
    """Write one JSON file per result into output_dir. Returns list of written paths."""
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for result, fname in zip(results, filenames):
        out_path = os.path.join(output_dir, fname)
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route('/annotate', methods=['POST'])
def annotate():
    """Annotate a single text or a list of texts.

    Request body (all fields except text/texts are required):
        text    : str                          — single text (returns one object)
        texts   : list[str | {text, metadata?}] — multiple texts (returns array)
        lang    : str                          — language code (e.g. "es")
        method  : str                          — pipeline method
        entities: list[str]                   — entity types to detect
        negation: bool  (default false)        — negation detection (biencoder only)
        output_mode: "return" | "directory"   (default "return")
        output_dir : str                       — required when output_mode="directory"

    output_mode="return"    → JSON response (single object or array)
    output_mode="directory" → writes text_NNNN.json files; returns summary dict
    """
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    params, err = _extract_pipeline_params(data)
    if err:
        return jsonify({"error": err}), 400

    # Accept 'text' (singular) or 'texts' (list)
    single = False
    if 'text' in data:
        raw_texts = [data['text']]
        single = True
    elif 'texts' in data:
        raw_texts = data['texts']
    else:
        return jsonify({"error": "Missing required field: 'text' (single) or 'texts' (list)"}), 400

    if not isinstance(raw_texts, list) or len(raw_texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400

    texts, metadatas, err = _normalize_texts(raw_texts)
    if err:
        return jsonify({"error": err}), 400

    pipeline = _build_pipeline(**params)
    results = _run_pipeline(pipeline, texts, metadatas)

    output_mode = data.get('output_mode', 'return')

    if output_mode == 'directory':
        output_dir = data.get('output_dir')
        if not output_dir:
            return jsonify({"error": "'output_dir' is required when output_mode is 'directory'"}), 400
        filenames = [f"text_{i:04d}.json" for i in range(len(results))]
        written = _write_to_dir(results, output_dir, filenames)
        return jsonify({"output_dir": output_dir, "files_written": written, "count": len(written)})
    
    if output_mode != 'return':
        return jsonify({"error": f"Invalid output_mode '{output_mode}'. Valid: 'return', 'directory'"}), 400
    
    return jsonify(results[0] if single else results)


@app.route('/annotate_dir', methods=['POST'])
def annotate_dir():
    """Annotate all .txt files inside a server-side directory.

    Request body:
        input_dir : str         — absolute path to directory containing .txt files
        lang      : str
        method    : str
        entities  : list[str]
        negation  : bool  (default false)
        output_mode: "return" | "directory"  (default "return")
        output_dir : str        — required when output_mode="directory"

    output_mode="return"    → JSON dict {filename: result_object}
    output_mode="directory" → writes <stem>.json files; returns summary dict
    """
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    if 'input_dir' not in data:
        return jsonify({"error": "Missing required field: 'input_dir'"}), 400

    input_dir = data['input_dir']
    if not os.path.isdir(input_dir):
        return jsonify({"error": f"'input_dir' does not exist or is not a directory: {input_dir}"}), 400

    params, err = _extract_pipeline_params(data)
    if err:
        return jsonify({"error": err}), 400

    txt_files = sorted(Path(input_dir).glob('*.txt'))
    if not txt_files:
        return jsonify({"error": f"No .txt files found in: {input_dir}"}), 400

    texts = [p.read_text(encoding='utf-8') for p in txt_files]
    filenames = [p.name for p in txt_files]
    metadatas = [{"source_file": str(p)} for p in txt_files]

    pipeline = _build_pipeline(**params)
    results = _run_pipeline(pipeline, texts, metadatas)

    output_mode = data.get('output_mode', 'return')

    if output_mode == 'directory':
        output_dir = data.get('output_dir')
        if not output_dir:
            return jsonify({"error": "'output_dir' is required when output_mode is 'directory'"}), 400
        out_filenames = [Path(f).stem + '.json' for f in filenames]
        written = _write_to_dir(results, output_dir, out_filenames)
        return jsonify({"output_dir": output_dir, "files_written": written, "count": len(written)})
    
    if output_mode != 'return':
        return jsonify({"error": f"Invalid output_mode '{output_mode}'. Valid: 'return', 'directory'"}), 400

    return jsonify(dict(zip(filenames, results)))


if __name__ == '__main__':
    app.run(debug=True)
