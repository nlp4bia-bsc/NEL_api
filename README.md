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
        text: "Este es un texto de ejemplo.\ncon un paciente procedente de almería aunque nacido en guadalupe, méxico, con mucha tos, mocos, fiebre y la varicela con meningitis.",
        footer: $footer
      },
      {
        text: "Otro texto con covid y paracetamol para probar.\ncon más  muchos más síntomas interesantes como edemas y negaciones como que 100% no tiene gripe A",
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
        "nlp_output": {
            "annotations": [
                {
                    "concept_class": "symptom",
                    "concept_code": "49727002",
                    "concept_confidence": 1.0,
                    "concept_str": "tos",
                    "controlled_vocabulary_concept_identifier": null,
                    "controlled_vocabulary_concept_official_term": null,
                    "controlled_vocabulary_namespace": null,
                    "controlled_vocabulary_source": null,
                    "controlled_vocabulary_version": null,
                    "dt4h_concept_identifier": null,
                    "end_offset": 116,
                    "extraction_confidence": 0.9999,
                    "mention_string": "tos",
                    "negation": "no",
                    "negation_confidence": null,
                    "nel_component_type": null,
                    "nel_component_version": null,
                    "ner_component_type": null,
                    "ner_component_version": null,
                    "start_offset": 113,
                    "uncertainty": "no",
                    "uncertainty_confidence": null
                },
                {
                    "concept_class": "symptom",
                    "concept_code": "301291007",
                    "concept_confidence": 0.7996,
                    "concept_str": "esputo acuoso",
                    "controlled_vocabulary_concept_identifier": null,
                    "controlled_vocabulary_concept_official_term": null,
                    "controlled_vocabulary_namespace": null,
                    "controlled_vocabulary_source": null,
                    "controlled_vocabulary_version": null,
                    "dt4h_concept_identifier": null,
                    "end_offset": 123,
                    "extraction_confidence": 0.9999,
                    "mention_string": "mocos",
                    "negation": "no",
                    "negation_confidence": null,
                    "nel_component_type": null,
                    "nel_component_version": null,
                    "ner_component_type": null,
                    "ner_component_version": null,
                    "start_offset": 118,
                    "uncertainty": "no",
                    "uncertainty_confidence": null
                },
                {
                    "concept_class": "symptom",
                    "concept_code": "64882008",
                    "concept_confidence": 1.0,
                    "concept_str": "fiebre",
                    "controlled_vocabulary_concept_identifier": null,
                    "controlled_vocabulary_concept_official_term": null,
                    "controlled_vocabulary_namespace": null,
                    "controlled_vocabulary_source": null,
                    "controlled_vocabulary_version": null,
                    "dt4h_concept_identifier": null,
                    "end_offset": 131,
                    "extraction_confidence": 0.9999,
                    "mention_string": "fiebre",
                    "negation": "no",
                    "negation_confidence": null,
                    "nel_component_type": null,
                    "nel_component_version": null,
                    "ner_component_type": null,
                    "ner_component_version": null,
                    "start_offset": 125,
                    "uncertainty": "no",
                    "uncertainty_confidence": null
                },
                {
                    "concept_class": "disorder/disease",
                    "concept_code": "10491005",
                    "concept_confidence": 0.9326,
                    "concept_str": "meningitis por varicela z\u00f3ster",
                    "controlled_vocabulary_concept_identifier": null,
                    "controlled_vocabulary_concept_official_term": null,
                    "controlled_vocabulary_namespace": null,
                    "controlled_vocabulary_source": null,
                    "controlled_vocabulary_version": null,
                    "dt4h_concept_identifier": null,
                    "end_offset": 160,
                    "extraction_confidence": 0.9999,
                    "mention_string": "varicela con meningitis",
                    "negation": "no",
                    "negation_confidence": null,
                    "nel_component_type": null,
                    "nel_component_version": null,
                    "ner_component_type": null,
                    "ner_component_version": null,
                    "start_offset": 137,
                    "uncertainty": "no",
                    "uncertainty_confidence": null
                }
            ],
            "processing_success": true,
            "record_metadata": {
                "admission_date": "2026-02-25T10:43:12.173783",
                "admission_id": "2",
                "admission_type": "emergency",
                "clinical_site_id": null,
                "deidentification_pipeline_name": null,
                "deidentification_pipeline_version": null,
                "deidentified": null,
                "nlp_processing_date": "2026-02-25T11:12:57.846757",
                "nlp_processing_pipeline_name": "NERL_Orchestrator",
                "nlp_processing_pipeline_version": "1.0",
                "patient_id": "1",
                "record_character_encoding": null,
                "record_creation_date": null,
                "record_extraction_date": null,
                "record_format": "txt",
                "record_id": "3",
                "record_lastupdate_date": null,
                "record_type": "discharge summary",
                "report_language": null,
                "report_section": null,
                "text": "Este es un texto de ejemplo.\ncon un paciente procedente de almer\u00eda aunque nacido en guadalupe, m\u00e9xico, con mucha tos, mocos, fiebre y la varicela con meningitis."
            }
        },
        "nlp_service_info": {
            "service_app_name": "DT4H NLP Processor",
            "service_language": "en",
            "service_model": "NERL_Orchestrator",
            "service_version": "1.0"
        }
    },
    {
        "nlp_output": {
            "annotations": [
                {
                    "concept_class": "disorder/disease",
                    "concept_code": "1119302008",
                    "concept_confidence": 0.7942,
                    "concept_str": "COVID-19 agudo",
                    "controlled_vocabulary_concept_identifier": null,
                    "controlled_vocabulary_concept_official_term": null,
                    "controlled_vocabulary_namespace": null,
                    "controlled_vocabulary_source": null,
                    "controlled_vocabulary_version": null,
                    "dt4h_concept_identifier": null,
                    "end_offset": 20,
                    "extraction_confidence": 0.9999,
                    "mention_string": "covid",
                    "negation": "no",
                    "negation_confidence": null,
                    "nel_component_type": null,
                    "nel_component_version": null,
                    "ner_component_type": null,
                    "ner_component_version": null,
                    "start_offset": 15,
                    "uncertainty": "no",
                    "uncertainty_confidence": null
                },
                {
                    "concept_class": "symptom",
                    "concept_code": "79654002",
                    "concept_confidence": 0.9147,
                    "concept_str": "edema",
                    "controlled_vocabulary_concept_identifier": null,
                    "controlled_vocabulary_concept_official_term": null,
                    "controlled_vocabulary_namespace": null,
                    "controlled_vocabulary_source": null,
                    "controlled_vocabulary_version": null,
                    "dt4h_concept_identifier": null,
                    "end_offset": 101,
                    "extraction_confidence": 0.9999,
                    "mention_string": "edemas",
                    "negation": "no",
                    "negation_confidence": null,
                    "nel_component_type": null,
                    "nel_component_version": null,
                    "ner_component_type": null,
                    "ner_component_version": null,
                    "start_offset": 95,
                    "uncertainty": "no",
                    "uncertainty_confidence": null
                },
                {
                    "concept_class": "symptom",
                    "concept_code": "58326007",
                    "concept_confidence": 0.7388,
                    "concept_str": "negativismo",
                    "controlled_vocabulary_concept_identifier": null,
                    "controlled_vocabulary_concept_official_term": null,
                    "controlled_vocabulary_namespace": null,
                    "controlled_vocabulary_source": null,
                    "controlled_vocabulary_version": null,
                    "dt4h_concept_identifier": null,
                    "end_offset": 114,
                    "extraction_confidence": 0.9997,
                    "mention_string": "negaciones",
                    "negation": "no",
                    "negation_confidence": null,
                    "nel_component_type": null,
                    "nel_component_version": null,
                    "ner_component_type": null,
                    "ner_component_version": null,
                    "start_offset": 104,
                    "uncertainty": "no",
                    "uncertainty_confidence": null
                },
                {
                    "concept_class": "disorder/disease",
                    "concept_code": "442438000",
                    "concept_confidence": 0.9132,
                    "concept_str": "gripe causada por virus Influenza A",
                    "controlled_vocabulary_concept_identifier": null,
                    "controlled_vocabulary_concept_official_term": null,
                    "controlled_vocabulary_namespace": null,
                    "controlled_vocabulary_source": null,
                    "controlled_vocabulary_version": null,
                    "dt4h_concept_identifier": null,
                    "end_offset": 145,
                    "extraction_confidence": 0.9998,
                    "mention_string": "gripe A",
                    "negation": "yes",
                    "negation_confidence": 0.9666,
                    "nel_component_type": null,
                    "nel_component_version": null,
                    "ner_component_type": null,
                    "ner_component_version": null,
                    "start_offset": 138,
                    "uncertainty": "no",
                    "uncertainty_confidence": null
                }
            ],
            "processing_success": true,
            "record_metadata": {
                "admission_date": "2026-02-25T10:43:12.173783",
                "admission_id": "2",
                "admission_type": "emergency",
                "clinical_site_id": null,
                "deidentification_pipeline_name": null,
                "deidentification_pipeline_version": null,
                "deidentified": null,
                "nlp_processing_date": "2026-02-25T11:12:57.846757",
                "nlp_processing_pipeline_name": "NERL_Orchestrator",
                "nlp_processing_pipeline_version": "1.0",
                "patient_id": "1",
                "record_character_encoding": null,
                "record_creation_date": null,
                "record_extraction_date": null,
                "record_format": "txt",
                "record_id": "3",
                "record_lastupdate_date": null,
                "record_type": "discharge summary",
                "report_language": null,
                "report_section": null,
                "text": "Otro texto con covid y paracetamol para probar.\ncon m\u00e1s  muchos m\u00e1s s\u00edntomas interesantes como edemas y negaciones como que 100% no tiene gripe A"
            }
        },
        "nlp_service_info": {
            "service_app_name": "DT4H NLP Processor",
            "service_language": "en",
            "service_model": "NERL_Orchestrator",
            "service_version": "1.0"
        }
    }
]
```
## License

MIT
