import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import json

from app.pipeline.orchestrator import NERL_Orchestrator

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

    texts = [
        "este es un texto de ejemplo.\ncon un paciente procedente de almería aunque nacido en guadalupe, méxico, con mucha tos, mocos, fiebre y la varicela con meningitis.",
        "otro texto con covid y paracetamol para probar.\ncon más  muchos más síntomas interesantes como edemas."
    ]

    orch = NERL_Orchestrator(agg_strat='first', device='cpu')
    
    results = orch.predict(texts, [random_footer, random_footer])
    for res in results:
        print(json.dumps(res, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    sys.exit(main())