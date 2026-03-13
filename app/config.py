import os

OBLIG_PROPERTIES = [
    "patient_id", "record_id", "admission_id", "admission_date", "admission_type"
]
DT4H_LANGS = [
    'es', 'en', 'it', 'ro', 'cz', 'se', 'nl'
]

AVAILABLE_ENTITIES = ["disease", "symptom", "pharmac"]

# Base directory where all models are stored.
# In Docker this will be a named volume mounted at /models.
# Locally it defaults to ./models for development convenience.
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/home/bscuser/Development/cogstack/ner-linking-api/app/resources/models")
GAZETTEER_CHACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/home/bscuser/Development/cogstack/ner-linking-api/app/resources/gazetteers")
VECTOR_DB_CHACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/home/bscuser/Development/cogstack/ner-linking-api/app/resources/vectorized_dbs")
REGISTRY_PATH = os.environ.get("MODEL_CACHE_DIR", "/home/bscuser/Development/cogstack/ner-linking-api/app/models/registry.yaml")


