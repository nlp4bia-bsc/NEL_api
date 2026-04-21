# Named Entity Recognition + Linking API

A Flask REST API that chains **Named Entity Recognition (NER)**, **Named Entity Linking (NEL)**, and optional **negation/uncertainty detection** into a single inference pipeline for clinical text. The API is model-agnostic: any Hugging Face token-classification model can be plugged in for NER, and multiple NEL backends are supported out of the box.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Validation](#validation)
5. [Running the Server](#running-the-server)
6. [API Reference](#api-reference)
   - [GET /](#get-)
   - [POST /annotate](#post-annotate)
   - [POST /annotate\_dir](#post-annotate_dir)
7. [Response Schema](#response-schema)
8. [Examples](#examples)
9. [Docker](#docker)
10. [Architecture](#architecture)

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU is strongly recommended. CPU-only mode works but is significantly slower for biencoder inference and vector DB construction.

---

## Installation

```bash
pip install uv
uv sync
```

---

## Configuration

All model and resource paths are managed through a **registry YAML file**. The path to this file is set in `app/config.py`:

```python
# app/config.py
REGISTRY_PATH = "app/model_manager/registry.yaml"
RESOURCES_PATH = "app/model_manager/resources"
```

### Registry structure

The registry maps languages and entity types to local model paths and gazetteer files. Below is a complete example for Spanish (`es`) with disease and symptom entities:

```yaml
ner:
  es:
    disease:
      repo_id: BSC-NLP4BIA/bsc-bio-ehr-es-carmen-distemist
      local_path: null   # will be populated after first download
    symptoms:
      repo_id: BSC-NLP4BIA/bsc-bio-ehr-es-carmen-symptemist
      local_path: null
    negation:
      repo_id: BSC-NLP4BIA/negation-tagger
      local_path: null

nel:
  es:
    repo_id: ICB-UMA/ClinLinker-KB-GP
    local_path: null

gazetteers:
  es:
    disease: /absolute/path/to/distemist_gazetteer.tsv
    symptoms: /absolute/path/to/symptemist_gazetteer.tsv

vectorized_dbs:
  es:
    disease: null   # built automatically on first run
    symptoms: null
```

**Key points:**

- **NER models** are per language and per entity type. The `negation` entry is required only when `negation: true` is used in requests.
- **NEL model** is shared across all entity types within a language.
- **Gazetteers** must be placed manually. Each must be a TSV file with at minimum a `term` column and a `code` column.
- **Vector databases** are built automatically from the gazetteer + NEL model on the first request. Once built, the path is written back to the registry so subsequent startups skip the build step. To force a rebuild, set the relevant entry to `null` in the registry.
- If a model already exists locally (e.g. pre-downloaded or manually placed), set `local_path` directly and leave `repo_id: null` — no download will be attempted.
- Swapping the NEL model produces a new vector DB filename automatically, triggering a rebuild.

### Device selection

The API detects CUDA availability at startup and sets the device accordingly. No manual configuration is needed.

---

## Validation

Before hosting the service, run the standalone test script to verify that all models load correctly and the full pipeline produces expected outputs:

```bash
uv run test_init.py
```

This also pre-builds any missing vector databases, which is recommended before the first real request (GPU strongly preferred for this step).

---

## Running the Server

```bash
uv run flask run --host=0.0.0.0 --port=5000
```

The health endpoint confirms the service is up:

```bash
curl http://localhost:5000/
# → OK
```

---

## API Reference

### `GET /`

Health check. Returns `200 OK` with body `OK`.

---

### `POST /annotate`

Annotate a **single text** or a **list of texts**.

#### Request body

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | `string` | one of `text`/`texts` | Single input text. Response is a single object (not an array). |
| `texts` | `array` | one of `text`/`texts` | List of texts. Each item may be a plain string or `{"text": "...", "metadata": {...}}`. Mixed lists are accepted. |
| `lang` | `string` | yes | Language code (e.g. `"es"`). |
| `method` | `string` | yes | NEL backend. See [Methods](#methods) below. |
| `entities` | `array[string]` | yes | Entity types to detect (e.g. `["disease", "symptoms"]`). Must match registry entries. |
| `negation` | `bool` | no | Enable negation/uncertainty detection (default: `false`). Only applies to `biencoder`. Requires a `negation` NER model in the registry. |
| `output_mode` | `string` | no | `"return"` (default) or `"directory"`. |
| `output_dir` | `string` | when `output_mode="directory"` | Absolute path to an existing or creatable output directory. Writes one `text_NNNN.json` file per input text. |

#### Methods

| Method | Description |
|---|---|
| `biencoder` | NER → dense retrieval NEL via sentence-transformer embeddings. Recommended for best accuracy. |
| `bm25` | NER → BM25 Okapi ranking over the gazetteer. |
| `levenshtein` | NER → fuzzy string matching (edit distance). |
| `jaro-winkler` | NER → fuzzy string matching (Jaro-Winkler similarity). |
| `token-sort-ratio` | NER → fuzzy token sort ratio matching. |
| `token-set-ratio` | NER → fuzzy token set ratio matching. |
| `lookup` | Direct dictionary lookup against the gazetteer. No NER step. |

#### Response

- `output_mode="return"`, `text` field: single result object.
- `output_mode="return"`, `texts` field: array of result objects.
- `output_mode="directory"`: summary object (see below).

```json
{
  "output_dir": "/path/to/output",
  "files_written": ["/path/to/output/text_0000.json", "..."],
  "count": 2
}
```

---

### `POST /annotate_dir`

Annotate all `.txt` files in a **server-side directory**.

#### Request body

| Field | Type | Required | Description |
|---|---|---|---|
| `input_dir` | `string` | yes | Absolute path to a directory containing `.txt` files. |
| `lang` | `string` | yes | Language code. |
| `method` | `string` | yes | NEL backend (see Methods table above). |
| `entities` | `array[string]` | yes | Entity types to detect. |
| `negation` | `bool` | no | Enable negation/uncertainty detection (default: `false`). |
| `output_mode` | `string` | no | `"return"` (default) or `"directory"`. |
| `output_dir` | `string` | when `output_mode="directory"` | Output directory. Each input `name.txt` produces `name.json`. |

#### Response

- `output_mode="return"`: JSON object keyed by filename, e.g. `{"nota_001.txt": {...}, "nota_002.txt": {...}}`.
- `output_mode="directory"`: summary object identical to the one from `/annotate`.

---

## Response Schema

Each result object returned by the API has the following structure:

```json
{
  "metadata": {
    "text": "<original input text>",
    "<key>": "<value>"
  },
  "annotations": [
    {
      "start":            0,
      "end":              6,
      "span":             "fiebre",
      "ner_class":        "ENFERMEDAD",
      "ner_score":        0.9999,
      "code":             "386661006",
      "term":             "fiebre",
      "nel_score":        1.0,
      "is_negated":       false,
      "negation_score":   0.0,
      "is_uncertain":     false,
      "uncertainty_score": 0.0
    }
  ],
  "processing_success": true,
  "processing_date": "21/04/2026, 14:32:01"
}
```

- `metadata` contains the original text and any per-item `metadata` dict supplied in the request (pass-through).
- `annotations` is empty (`[]`) when no entities are detected.
- Character offsets (`start`/`end`) refer to the original unmodified input text.

---

## Examples

Start the server before running any of the examples below:

```bash
uv run flask run --host=0.0.0.0 --port=5000
```

### Health check

```bash
curl http://localhost:5000/
```

### Single text

```bash
curl -X POST http://localhost:5000/annotate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "El paciente presenta fiebre alta y tos persistente.",
    "lang": "es",
    "method": "biencoder",
    "entities": ["disease", "symptoms"]
  }'
```

### Bulk texts

```bash
curl -X POST http://localhost:5000/annotate \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      "El paciente presenta fiebre alta y tos persistente.",
      "Dolor abdominal agudo. No presenta náuseas."
    ],
    "lang": "es",
    "method": "biencoder",
    "entities": ["disease", "symptoms"]
  }'
```

### Bulk texts with per-item metadata

```bash
curl -X POST http://localhost:5000/annotate \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      {
        "text": "El paciente presenta fiebre alta.",
        "metadata": {"patient_id": "1", "record_id": "A01"}
      },
      {
        "text": "Dolor abdominal agudo sin náuseas.",
        "metadata": {"patient_id": "2", "record_id": "A02"}
      }
    ],
    "lang": "es",
    "method": "biencoder",
    "entities": ["disease", "symptoms"]
  }'
```

The `metadata` dict is merged into the `metadata` field of each result object.

### With negation detection

```bash
curl -X POST http://localhost:5000/annotate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "El paciente no presenta fiebre ni tos.",
    "lang": "es",
    "method": "biencoder",
    "entities": ["disease", "symptoms"],
    "negation": true
  }'
```

Entities in a negated context will have `"is_negated": true` and a non-zero `negation_score`.

### Save results to disk

```bash
curl -X POST http://localhost:5000/annotate \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": ["Fiebre alta.", "Tos seca."],
    "lang": "es",
    "method": "biencoder",
    "entities": ["disease", "symptoms"],
    "output_mode": "directory",
    "output_dir": "/path/to/output"
  }'
```

Writes `/path/to/output/text_0000.json` and `text_0001.json`. Returns:

```json
{"count": 2, "files_written": ["/path/to/output/text_0000.json", "/path/to/output/text_0001.json"], "output_dir": "/path/to/output"}
```

### Annotate a directory of .txt files

```bash
curl -X POST http://localhost:5000/annotate_dir \
  -H 'Content-Type: application/json' \
  -d '{
    "input_dir": "/path/to/corpus",
    "lang": "es",
    "method": "biencoder",
    "entities": ["disease", "symptoms"]
  }'
```

Returns a JSON object keyed by filename:

```json
{
  "nota_001.txt": { "metadata": {...}, "annotations": [...], ... },
  "nota_002.txt": { "metadata": {...}, "annotations": [...], ... }
}
```

To mirror the corpus as annotated JSON files:

```bash
curl -X POST http://localhost:5000/annotate_dir \
  -H 'Content-Type: application/json' \
  -d '{
    "input_dir": "/path/to/corpus",
    "lang": "es",
    "method": "biencoder",
    "entities": ["disease", "symptoms"],
    "output_mode": "directory",
    "output_dir": "/path/to/annotated"
  }'
```

### Running the test suite

```bash
# Start the server first, then in another terminal:
bash test_endpoints.sh

# Against a non-default host:
bash test_endpoints.sh http://hostname:5000
```

---

## Docker

```bash
docker compose up
```

Or for a clean rebuild:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

By default the container runs without CUDA. To enable GPU support, edit `docker-compose.yml`:

```yaml
environment:
  CUDA_VISIBLE_DEVICES: "0"   # set to your GPU index
```

The default port is `5000` and can be changed in `docker-compose.yml`.

---

## Architecture

```
POST /annotate
      │
      ▼
 Input validation & text normalisation
      │
      ▼
 Pipeline instantiation
 (method × lang × entities → LocalResolver → model paths)
      │
      ▼
 NER  — HuggingFace token-classification model
  │        sentence splitting + max-length chunking
  │        → list of {start, end, span, ner_class, ner_score}
  │
  ▼
 NEL  — backend selected by `method`
  │        biencoder: SentenceTransformer query embedding
  │                   → cosine similarity over pre-built vector DB
  │        bm25 / fuzzy: lexical matching over gazetteer TSV
  │        lookup: direct dictionary lookup
  │        → adds {code, term, nel_score} to each annotation
  │
  ▼
 Negation (optional)
  │        dedicated NER model produces NEG/NSCO/UNC/USCO spans
  │        → overlap detection adds {is_negated, is_uncertain, ...}
  │
  ▼
 Post-processing
        merge contiguous same-class entities, deduplicate, sort by offset
      │
      ▼
 DataFormatter.serialize()
        wraps into {metadata, annotations, processing_success, processing_date}
      │
      ▼
 JSON response  or  write to output_dir
```

### Key files

| Path | Role |
|---|---|
| `app/__init__.py` | Flask app, endpoints, shared helpers |
| `app/config.py` | Registry and resource base paths |
| `app/src/pipelines.py` | All pipeline implementations |
| `app/src/ner/` | Token classification inference |
| `app/src/nel/` | NEL backends (biencoder, bm25, fuzzy, lookup) |
| `app/src/negation/` | Negation/uncertainty attribution |
| `app/src/format/` | Output formatters |
| `app/model_manager/resolver.py` | Single source of truth for resource paths |
| `app/model_manager/registry.yaml` | Model and gazetteer path registry |
| `test_init.py` | Pre-flight pipeline validation |
| `test_endpoints.sh` | HTTP-level endpoint tests |

---

## License

MIT
