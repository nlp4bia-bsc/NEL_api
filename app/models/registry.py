"""
Model and Resource Registries

These registries define the resources available for each language and
entity type used in the pipeline. The pipeline components will dynamically
load models or resources based on these mappings.
"""
# Token-classification NER models
# Maps: language -> entity_type -> model_identifier_or_path
NER_REGISTRY: dict[str, dict[str, str]] = {
    "es": {
        "disease": "BSC-NLP4BIA/bsc-bio-ehr-es-carmen-distemist",
        "symptoms": "BSC-NLP4BIA/bsc-bio-ehr-es-carmen-symptemist",
    }
}

# Negation detection models
# Maps: language -> model_identifier_or_path
NEG_REGISTRY: dict[str, str] = {
    "es": "BSC-NLP4BIA/negation-tagger"
}

# Named Entity Linking (NEL) models (BiEncoder)
# Maps: language -> model_identifier_or_path
NEL_REGISTRY: dict[str, str] = {
    "es": "ICB-UMA/ClinLinker-KB-GP"
}

# Gazetteer resources used by the NEL component
# Maps: language -> entity_type -> resource_paths
#
# Notes:
# - `gaz_path` must point to the gazetteer file or directory.
# - `vector_db_path` must always be present as a key. The value may be an
#   empty string if the vector database has not been created yet, but the
#   key itself must not be omitted or set to None.
GAZ_REGISTRY: dict[str, dict[str, dict[str, str]]] = { # set this uppp
    "es": {
        "disease": {
            "gaz_path": "es/distemist/distemist_gazetteer.csv",
            "vector_db_path": "es/distemist/distemist_vector_db.pt",
        },
        "symptoms": {
            "gaz_path": "es/symptemist/symptemist_gazetteer.csv",
            "vector_db_path": "es/symptemist/symptemist_vector_db.pt",
        },
    }
}


# ---------------------------------------------------------------------------
# Registry validation utilities
# ---------------------------------------------------------------------------

def check_lang(lang: str) -> tuple[bool, list[str]]:
    """
    Validate that a language is correctly configured in the model registries.

    This function verifies that all required components for the SOTA pipeline
    are available for the given language:

    - Token classification NER models
    - Gazetteers for entity linking
    - Negation detection model
    - NEL (BiEncoder) model

    Parameters
    ----------
    lang : str
        Language code (e.g. "es", "en").

    Returns
    -------
    tuple[bool, list[str]]

        A tuple containing:

        1. bool
            True if the language has all required pipeline components
            (NER, gazetteer, negation model, and NEL model).

        2. list[str]
            List of entity types that have both:
            - a NER model
            - a gazetteer entry

            These entity types can be safely processed by the pipeline.

    Notes
    -----
    A language may still partially function if some components are missing.
    The returned entity list allows the pipeline to restrict execution to
    supported entity types.
    """

    supported_entities: list[str] = []

    ner_entities = set(NER_REGISTRY.get(lang, {}).keys())
    gaz_entities = set(GAZ_REGISTRY.get(lang, {}).keys())

    # entities supported by both NER and gazetteer
    supported_entities = sorted(list(ner_entities & gaz_entities))

    has_neg = lang in NEG_REGISTRY
    has_nel = lang in NEL_REGISTRY

    fully_supported = bool(supported_entities) and has_neg and has_nel

    return fully_supported, supported_entities