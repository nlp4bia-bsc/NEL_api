from flask import Flask, request, jsonify
from pipeline.orchestrator import run_nerl_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route('/nerl_process_bulk', methods=['POST'])
def nerl_process_bulk():
    """
    Process multiple texts using the NER models and normalizing to the provided gazetteers.
    ---
    parameters:
      - content: list
        - text: str
      - nerl_models_config: list (optional)
        - ner_model_path: str
        - gazetteer_path: str
    responses:
      200:
        description: Processed texts with NER annotations normalized.
    """
    body = request.json

    if not isinstance(body, dict) or not "content" in body.keys():
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400

    content = body["content"]

    if not isinstance(content, list):
        return jsonify({"error": "Input must be a list of objects"}), 400
    
    texts = []
    for item in content:
            text = item.get('text')
            if not text:
                return jsonify({"error": "Each item must contain 'text'"}), 400
            texts.append(text)

    results = run_nerl_pipeline(texts, agg_strat='first', device='cpu')
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
