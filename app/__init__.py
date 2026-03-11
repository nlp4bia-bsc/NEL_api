"""
Flask Application
=================
Request body shape for /process_bulk:

{
    "content": [
        {"text": "...", "footer": {...}},
        ...
    ],
    "args": {                                   # optional
        "language": "es",                       # language in ISO 639-1 format
        "entities": ["disease", "symptom"],     # list of entity types
        "method": "sota"                        # "sota" | "lookup" (default: "sota")
    }
}
"""

from flask import Flask, request, jsonify

from app.src.pipelines import SotaPipeline, LookupPipeline
from app.models.registry import check_lang
from app.config import OBLIG_PROPERTIES, DT4H_LANGS, AVAILABLE_ENTITIES

app = Flask(__name__)

method2pipeline = {
    "sota": lambda **kwargs: SotaPipeline(agg_strat="first", **kwargs),
    "lookup": LookupPipeline,
}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route("/process_bulk", methods=["POST"])
def process_bulk():
    """
    Process multiple texts using the configured annotation pipeline.
    """
    data = request.json

    if not isinstance(data, dict) or "content" not in data:
        return jsonify({"error": "Input must be a dictionary with a 'content' key"}), 400

    content = data["content"]
    if not isinstance(content, list):
        return jsonify({"error": "'content' must be a list of objects"}), 400

    # --- Parse args ---------------------------------------------------------- (CAN HANDLE THIS BETTER WHEN WE IMPLEMENT THE OTHER APPROACHES, WILL WORK FOR NOW)
    args = data.get("args", {}) or {}
    method: str = args.get("method", "sota")
    if method not in ("sota", "lookup"):
        return jsonify({"error": f"Unknown method {method!r}. Use 'sota' or 'lookup'."}), 400
    
    lang: str = args.get("language", "es")
    valid_lang, available_entities = check_lang(lang)
    if not valid_lang:
        return jsonify({"error": f"Language {lang!r} not fully implemented for sota pipeline."}), 400
    
    ents: list = args.get("entities", ["disease"])
    if any([ent not in available_entities for ent in ents]):
        incorrect_ent = [ent for ent in ents if ent not in available_entities]
        return jsonify({"error": f"The following entities are not considered in the registry either because the gazetteer or the ner model for that language is not specified{incorrect_ent!r}"}), 400

    # --- Validate content ----------------------------------------------------
    texts, footers = [], []
    for item in content:
        text = item.get("text")
        footer = item.get("footer")
        if not text:
            return jsonify({"error": "Each item must contain a non-empty 'text' field"}), 400
        missing_props = [p for p in OBLIG_PROPERTIES if p not in (footer or {})]
        if missing_props:
            return jsonify({
                "error": "Missing obligatory footer properties: " + ", ".join(missing_props)
            }), 400
        texts.append(text)
        footers.append(footer)

    # --- Run pipeline --------------------------------------------------------
    try:
        pipeline = method2pipeline[method](lang, ents)
    
        results = pipeline.predict(texts=texts)

    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Pipeline error")
        return jsonify({"error": "Internal pipeline error", "detail": str(exc)}), 500

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)