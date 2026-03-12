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
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/gpfs/projects/bsc14/erodrig/NEL_api/app/local_models")
GAZETTEER_CHACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/gpfs/projects/bsc14/erodrig/NEL_api/app/local_gazetteers")
VECTOR_DB_CHACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/gpfs/projects/bsc14/erodrig/NEL_api/app/local_vector_dbs")

LOOKUP_PATH = "/gpfs/projects/bsc14/erodrig/NEL_api/app/lookup.csv"