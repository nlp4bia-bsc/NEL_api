# Named Entity Recognition + Linking (NERL) Inference API

This is a Flask API that performs named entity recognition (NER) and linking/normalization (NEL) using any Hugging Face token classification model of your preference for NER and configuring the gazetters, and sentence similarity & text-reranking models for normalization.

## 💻 Prerequisites

⚠ This API will work much faster if it has access to a GPU, though it can also work with CPU only at a much slower pace.

Python 3.10 (at least) required, 3.13 recommended.

First, create a virtual environment, activate it, and install `requirements.txt`.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
```

### 🛠 (Recommended) Configure the default models and gazetteers

To avoid sending the `nerl_models_config` on each request, you can just configure the default models and gazetteers to be used in `app/config.py`. Note that model paths can also be Hugging Face identifiers (e.g. _BSC-NLP4BIA/bsc-bio-ehr-es-carmen-symptemist_). In such case, the models will be downloaded the first time and saved in local cache for future uses.

Anyway, if you provide them in the HTTP request, they will override the default ones specified.

## 🏃‍♀️ Running the service

To start the service, just run the `__init__.py` file.

```
python3 __init__.py
```

This will create a Flask API web server listening on port 5000 (or the one you defined). Take into account that it will be on _development mode_, meaning that it cannot handle multiple simultaneous requests correctly, it does not auto-restart, and we don't have TLS certificates so that it can be called using HTTPS, only HTTP.

We should get something like:

```
user@computer$ python __init__.py
 * Serving Flask app '__init__'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: XXX-YYY-ZZZ

```

## Available endpoints

These endpoints can be called from Python using its built-in `requests` package, any other HTTP client library from any programming language, `curl`, Postman, etc.

- `nerl_process_bulk`: perform named entity recognition and linking using the specified models and gazetteers (or default ones) to all texts provided. It works for multiple models, which will be run sequentially and their outputs combined, with mentions ordered by their order of appearance (offset) in text.

```
parameters:
      - content: list
        - text: str
      - nerl_models_config: list
        - ner_model_path: str (local path or HF identifier)
        - nel_model_path: str (local path or HF identifier)
        - gazetteer_path: str
        - vectorized_gazetteer_path: str (optional)
```

Example:

- Input

```
{
  "content":
    [
        {
            "text": "Paciente, de 1a 10m, es traído por su madre por fiebre de hasta 39ºC de 4 días de evolución con tos y abundante\nmucosidad nasal. No clínica gastrointestinal. Diuresis e ingesta conservadas. No otra clínica. Ambiente\nepidemiológico familiar negativo. Han administrado antitermia en domicilio, última toma ibuprofeno en urgencias\n(hace 30 min aprox). Última toma salbutamol, 3 pulsaciones a las 17 h.\nPositivo por gripe A el 10/01.\nHa realizado tratamiento con salbutamol inhalado desde 3-4/01 por broncoespasmo, y 7 días de prednisolona\noral (fin hace 1 semana). Consultas en atención primaria y H. del Mar, última ayer 13/01 en H. del Mar donde se\nadministra dexametasona a 0.3mg/kg por tos laríngea y se alta a domicilio."
        },
        {
            "text": "Paciente de 15 años que acude para intervención quirúrgica programada por presentar frenillo peneano corto."
        }
    ]
}
```

- Output

```
[
        {
            "code": "64882008",
            "end": 54,
            "is_negated": false,
            "is_uncertain": false,
            "negation_score": null,
            "nel_score": 1.0,
            "ner_class": "SINTOMA",
            "ner_score": 0.9999,
            "span": "fiebre",
            "start": 48,
            "term": "fiebre",
            "uncertainty_score": null
        },
        {
            "code": "49727002",
            "end": 99,
            "is_negated": false,
            "is_uncertain": false,
            "negation_score": null,
            "nel_score": 1.0,
            "ner_class": "SINTOMA",
            "ner_score": 0.9999,
            "span": "tos",
            "start": 96,
            "term": "tos",
            "uncertainty_score": null
        },
        {
            "code": "64531003",
            "end": 126,
            "is_negated": false,
            "is_uncertain": false,
            "negation_score": null,
            "nel_score": 0.9099,
            "ner_class": "SINTOMA",
            "ner_score": 0.9999,
            "span": "mucosidad nasal",
            "start": 111,
            "term": "secreción nasal",
            "uncertainty_score": null
        },
        {
            "code": "236060008",
            "end": 155,
            "is_negated": true,
            "is_uncertain": false,
            "negation_score": 0.9999,
            "nel_score": 0.6282,
            "ner_class": "SINTOMA",
            "ner_score": 0.9999,
            "span": "clínica gastrointestinal",
            "start": 131,
            "term": "trastornos sintomáticos del tracto gastrointestinal",
            "uncertainty_score": null
        },
      ...
    ]
```

## Docker Support (in progress)
