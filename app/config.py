NER_PATHS = [
    "app/resources/ner_models/carmen-core-models/bsc-bio-ehr-es-carmen-distemist", 
    "app/resources/ner_models/carmen-core-models/bsc-bio-ehr-es-carmen-symptemist",
    "app/resources/ner_models/negation" # NEGATION MUST BE THE LAST!!
]

NEL_PATHS = [
    (
        "app/resources/gazetteers/distemist_gazetteer.tsv",
        "app/resources/nel_models/ICB-UMA--ClinLinker-KB-GP-st",
        "app/resources/gazetteers/vectorized_distemist_gazetteer.pt"
    ),
    (
        "app/resources/gazetteers/symptemist_gazetteer.tsv",
        "app/resources/nel_models/ICB-UMA--ClinLinker-KB-GP-st",
        "app/resources/gazetteers/vectorized_symptemist_gazetteer.pt"
    )
]

OBLIG_PROPERTIES = ["patient_id", "record_id", "admission_id", "admission_date", "admission_type"]