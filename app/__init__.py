from functools import partial
from flask import Flask, request, jsonify
from app.src.pipelines import LookupPipeline, FuzzyMatchPipeline, BM25OkapiPipeline, BiencoderPipeline
from app.config import OBLIG_PROPERTIES


app = Flask(__name__)

method2pipeline = {
    'lookup': LookupPipeline,
    'levenshtein': partial(FuzzyMatchPipeline, method = 'levenshtein'),
    'jaro-winkler': partial(FuzzyMatchPipeline, method = 'jaro_winkler'),
    'token-sort-ratio': partial(FuzzyMatchPipeline, method = 'token_sort_ratio'),
    'token-set-ratio': partial(FuzzyMatchPipeline, method = 'token_set_ratio'),
    'bm25': BM25OkapiPipeline,
    'biencoder': partial(BiencoderPipeline, negation=True),
}

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route('/process_bulk', methods=['POST'])
def process_bulk():
    """
    Process multiple texts using NER Dictionary Lookup
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: bulk_input
          type: array
          items:
            type: object
            required:
              - text
            properties:
              text:
                type: string
                description: The text to process
    responses:
      200:
        description: Processed texts with NER annotations
    """
    data = request.json

    if not isinstance(data, dict):
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400

    if not "content" in data.keys():
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400
    
    method: str = 'biencoder'
    if isinstance(data.get("args", ""), str):
        if (mth:=data.get("args", "")) in method2pipeline.keys():
            method = mth

    data = data["content"]

    if not isinstance(data, list):
        return jsonify({"error": "Input must be a list of objects"}), 400

    texts = []
    footers = []
    for item in data:
        text = item.get('text')
        footer = item.get('footer')
        if not text:
            return jsonify({"error": "Each item must contain 'text'"}), 400
        if any([obligatory_prop not in footer for obligatory_prop in OBLIG_PROPERTIES]):
            return jsonify({"error": "The input has a missing obligatory property: " + ", ".join(OBLIG_PROPERTIES)}), 400

        texts.append(text)
        footers.append(footer)

    lang = 'es'
    entities = ["disease", "symptoms"]

    pipeline = method2pipeline[method](lang=lang, entities=entities)
    
    results = pipeline.predict(texts=texts)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
