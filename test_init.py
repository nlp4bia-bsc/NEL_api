import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import json
from app.src.pipelines import BiencoderPipeline, LookupPipeline
from app.src.format import PassthroughFormatter

method2pipeline = {
    "sota": lambda **kwargs: BiencoderPipeline(agg_strat="first", negation=True, **kwargs),
    "lookup": LookupPipeline,
}

cdm2formatter = {
    'none': PassthroughFormatter
}

def main():
    """
    this is just for testing purposes.
    """
    random_footer = {
        "patient_id": "1",
        "admission_id": "2",
        "admission_date": '2026-02-25T10:43:12.173783',
        "admission_type": "emergency",
        "record_id": "3",
        "record_type": "discharge summary",
        "record_format": "txt"
    }
    footers = [random_footer, random_footer]

    texts = [
        "Este es un texto de ejemplo.\ncon un paciente procedente de almería aunque nacido en guadalupe, méxico, con mucha tos, mocos, fiebre y la varicela con meningitis.",
        "Otro texto con covid y paracetamol para probar.\ncon más  muchos más síntomas interesantes como edemas y negaciones como que 100% no tiene gripe A."
    ]

    method = 'sota'
    lang = "es"
    entities = ["disease"]
    # entities = ["disease"]
    pipe = method2pipeline[method](lang=lang, entities=entities)

    cdm = 'none'
    formatter = cdm2formatter[cdm]()
    
    annotations = pipe.predict(texts)
    results = [formatter.serialize(text, ann, footer) for text, ann, footer in zip(texts, annotations, footers)]
    for res in results:
        print(json.dumps(res, ensure_ascii=False, indent=4))
    

if __name__ == "__main__":
    sys.exit(main())