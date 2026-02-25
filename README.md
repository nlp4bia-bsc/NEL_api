# Named Entity Recognition + Linking (NERL) Inference API

A Flask-based REST API that chains **Named Entity Recognition (NER)**, **Named Entity Linking/Normalization (NEL)**, and optional **negation detection** into a single inference pipeline. It is model-agnostic: any Hugging Face token classification model can be plugged in for NER, and any sentence-similarity or re-ranking model can be used for NEL. Multiple model configurations can run in the same request, with their outputs merged and ordered by text offset.

---

## Overview

The codebase is organized as follows:

- **`app/`** — Core application logic: Flask routes, model loading, NER/NEL inference, negation tagging, and gazetteer handling.
- **`app/config.py`** — Default model and gazetteer paths/identifiers. Edit this to avoid repeating configuration on every request.
- **`main.py`** — Entry point to start the Flask development server.
- **`test_init.py`** — Standalone test script to validate the full pipeline (NER → NEL → negation) before hosting the API. Run this first to make sure everything is wired correctly.
- **`sample_footer.json`** — Example footer payload showing how to configure models and gazetteers in a request.
- **`Dockerfile` / `docker-compose.yml`** — Container support (in progress).

---

## Requirements

- Python 3.10+ (3.13 recommended)
- A GPU is strongly recommended; CPU-only mode works but is significantly slower.

---

## Installation

```bash
pip install uv
uv init
uv sync
```

---

## Configuration

Before running the API, configure the default models and gazetteer in **`app/config.py`**. The negation model must be the last one in the ner models list.

You must install the ner models and gazetteers into your local directory, under whichever path you specify. In this example, we install the necessary models and gazetteers into app/test_resources/ like so.
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
mkdir -p app/test_resources/gazetteers app/test_resources/nel_models app/test_resources/ner_models
```
```bash
mkdir -p app/test_resources/ner_models/distemist app/test_resources/ner_models/negation app/test_resources/nel_models/clinlinker
hf download BSC-NLP4BIA/bsc-bio-ehr-es-carmen-distemist --local-dir app/test_resources/ner_models/distemist
hf download BSC-NLP4BIA/negation-tagger --local-dir app/test_resources/ner_models/negation
hf download ICB-UMA/ClinLinker-KB-GP --local-dir app/test_resources/nel_models/clinlinker
```
Finally, download the gazetteer of your choice. In our case, continuing with focus on Distemist, we will use the one [here](https://zenodo.org/records/6458115), and place it in the gazetteer directory. If you followed this example, it should have this structure:
```bash
.
├── gazetteers
│   └── dictionary_distemist.tsv
├── nel_models
│   └── clinlinker
│       ├── config.json
│       ├── merges.txt
│       ├── pytorch_model.bin
│       ├── README.md
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.json
└── ner_models
    ├── distemist
    │   ├── config.json
    │   ├── merges.txt
    │   ├── pytorch_model.bin
    │   ├── README.md
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   ├── tokenizer.json
    │   ├── training_args.bin
    │   └── vocab.json
    └── negation
        ├── config.json
        ├── merges.txt
        ├── pytorch_model.bin
        ├── README.md
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        ├── training_args.bin
        └── vocab.json
```
Though in theory you can have any placement as long it is within the `app/` directory (for the subsequent dockerization) and correctly referenced in the `config.py` file

### Example: Disease NER + Linking (DisTEMIST)

```python
# app/config.py

NER_PATHS = [
    "app/test_resources/ner_models/distemist", 
    "app/test_resources/ner_models/negation" # NEGATION MUST BE THE LAST!!
]

NEL_PATHS = [
    (
        "app/test_resources/gazetteers/dictionary_distemist.tsv",
        "app/test_resources/nel_models/clinlinker",
        "app/test_resources/gazetteers/vectorized_gazetteer_distemist.pt" # if this file doesn't exist, it will be created at runtime
    )
]

OBLIG_PROPERTIES = <list of metadata that the output must contain>
```

On the first request, the API will automatically build a **vector database** from the gazetteer entries and saves it for fast retrieval on subsequent runs. It is recommended to use GPU's for this process at least since it is the most computationally expensive.

> **Note:** The gazetteer has to be a *.tsv* file with at least the columns `term` and `code` in order to be able to create the vector db and use it in the bi-encoder.
---

## Validate before hosting: 

Before hosting the service, we can debug and check everything is running smoothly. To do so, go to the directory where you cloned the repository and do the following tests.

### Running `test_init.py`

Before starting the API server, run the provided test script to verify that all models load correctly and the full NER → NEL → negation pipeline produces expected outputs:

```bash
uv run test_init.py
```

This is the recommended sanity check before exposing the service. Also, if you haven't yet created the vector database of the gazetteers you'll be using, it will create them instead of waiting for them to be created on an api call.

---

### Running the API

```bash
uv run flask run--host=0.0.0.0--port=5000
```

The server starts on `http://127.0.0.1:5000` in development mode, which is all we need since we will be calling the API from our own machine.

You can test how well your API is working calling the endpoints specified down in [here](#API Endpoint)

## Building:

The docker files are already created and ready to use. You only have to execute the following commands:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```
By default, the API won't use your CUDA environment to avoid hardware incompatibility issues, but you can change it from the `docker-compose.yml` file:

```yaml
environment:
      CUDA_VISIBLE_DEVICES: "" // by default, but change to "0" if you have cuda-compatible GPU's
```

Also, the port is set to `5000` but it can also be modified freely at any moment by changing both the `docker-compose.yml` or `Dockerfile` files.

---

## API Endpoint

### `POST /process_bulk`

Runs the full NERL pipeline over a list of texts. Each item in `content` contains a `text` field and a `footer` that carries the model/gazetteer configuration.

**Request body:**
```json
{
  "content": [
    {
      "text": "<clinical text>",
      "footer": { <rest of variables to be output along with the annotations> }
    }
  ]
}
```
---

## Example Request

Using `sample_footer.json` (which contains the model and gazetteer configuration), send two clinical sentences for processing:

```bash
curl --location 'http://127.0.0.1:5000/process_bulk' \
--header 'Content-Type: application/json' \
--data "$(jq -n \
  --argjson footer "$(cat sample_footer.json)" \
  '{
    content: [
      {
        text: "A 74-year-old woman was admitted to our hospital because of dyspnea and chest pain for 1 month.",
        footer: $footer
      },
      {
        text: "A 6-year-old man was admitted to our hospital because of metastatic tumor.",
        footer: $footer
      }
    ]
  }'
)"
```

> **Note:** `jq` must be installed. The `sample_footer.json` file in the repository root contains a ready-to-use configuration example.

### Example Response

Each detected entity is returned as an object with NER, NEL, and negation fields:

```json
[
  {
    "span": "dyspnea",
    "start": 58,
    "end": 65,
    "ner_class": "DISEASE",
    "ner_score": 0.9997,
    "code": "267036007",
    "term": "dyspnea",
    "nel_score": 1.0,
    "is_negated": false,
    "is_uncertain": false,
    "negation_score": null,
    "uncertainty_score": null
  },
  {
    "span": "chest pain",
    "start": 70,
    "end": 80,
    "ner_class": "DISEASE",
    "ner_score": 0.9995,
    "code": "29857009",
    "term": "chest pain",
    "nel_score": 0.9871,
    "is_negated": false,
    "is_uncertain": false,
    "negation_score": null,
    "uncertainty_score": null
  },
  {
    "span": "metastatic tumor",
    "start": 55,
    "end": 71,
    "ner_class": "DISEASE",
    "ner_score": 0.9991,
    "code": "128462008",
    "term": "metastatic neoplasm",
    "nel_score": 0.8743,
    "is_negated": false,
    "is_uncertain": false,
    "negation_score": null,
    "uncertainty_score": null
  }
]
```
## License

MIT
