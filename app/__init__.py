from flask import Flask, request, jsonify
from app.pipeline.orchestrator import NERL_Orchestrator
from app.config import OBLIG_PROPERTIES

app = Flask(__name__)

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
    
    orch = NERL_Orchestrator(agg_strat='first', device='cpu')
    results = orch.predict(texts=texts, footers=footers)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
