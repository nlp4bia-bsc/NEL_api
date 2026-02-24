NER_PATHS = [
    "resources/ner_models/carmen-core-models/bsc-bio-ehr-es-carmen-distemist", 
    "resources/ner_models/carmen-core-models/bsc-bio-ehr-es-carmen-symptemist",
    "resources/ner_models/negation" # NEGATION MUST BE THE LAST!!
]

NEL_PATHS = [
    (
        "resources/gazetteers/distemist_gazetteer.tsv",
        "resources/nel_models/ICB-UMA--ClinLinker-KB-GP-st",
        "resources/gazetteers/vectorized_distemist_gazetteer.pt"
    ),
    (
        "resources/gazetteers/symptemist_gazetteer.tsv",
        "resources/nel_models/ICB-UMA--ClinLinker-KB-GP-st",
        "resources/gazetteers/vectorized_symptemist_gazetteer.pt"
    )
]