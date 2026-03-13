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

REGISTRY_PATH = "app/default_registry.yaml"