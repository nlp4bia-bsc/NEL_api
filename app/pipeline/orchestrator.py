import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pipeline.ner import ner_inference
from pipeline.nel import nel_inference
from pipeline.negation import add_negation_uncertainty_attributes
from config import NER_PATHS, NEL_PATHS

def run_nerl_pipeline(texts: list[str], agg_strat: str = "first", device: str | None = None) -> list:
    ner_results = ner_inference(texts, NER_PATHS, agg_strat=agg_strat, device=device)
    neg_results = ner_results.pop()
    norm_results = nel_inference(ner_results, NEL_PATHS, combined=True, device=device)
    final_results = add_negation_uncertainty_attributes(norm_results, neg_results)
    return final_results

import sys

def main():
    """
    this is just for testing purposes.
    """
    texts = [
        "este es un texto de ejemplo.\ncon un paciente procedente de almería aunque nacido en guadalupe, méxico, con mucha tos, mocos, fiebre y la varicela con meningitis.",
        "otro texto con covid y paracetamol para probar.\ncon más  muchos más síntomas interesantes como edemas."
    ]
    
    results = run_nerl_pipeline(texts=texts, device='cpu')
    print(results)


if __name__ == "__main__":
    sys.exit(main())