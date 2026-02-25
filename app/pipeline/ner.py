"""
simple_inference.py

This script performs NER (i.e., token classification) inference on a set of text files using a HuggingFace Transformers model.
It reads .txt files, runs the model, and writes .ann annotation files in BRAT format.

Usage example:
  python simple_inference.py -i <input_txt_dir> -o <output_ann_dir> -m <model_path> [--overwrite] [--agg_strat <strategy>]

Author: Jan Rodríguez Miret
"""
import torch
from transformers import pipeline
from spacy.lang.es import Spanish
from pathlib import Path

from app.utils.text_preprocessing import pretokenize_sentence
from app.utils.results_postprocessing import join_all_entities, align_results

class NerModel:
    def __init__(self, model_checkpoint: str | Path, agg_strat: str = "first", device: str | None = None):
        self.nlp = Spanish()
        self.nlp.add_pipe("sentencizer")
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.pipe = pipeline(
            task="token-classification", 
            model=model_checkpoint, 
            aggregation_strategy=agg_strat, 
            device=self.device,
            )

    def _process_sentence(self, sentence: str, sentence_start_offset: int) -> list[dict]:
        # Pretokenize sentence for model compatibility
        sentence_pretokenized, added_spaces_pos = pretokenize_sentence(sentence)
        # Run model inference
        results_pre = self.pipe(sentence_pretokenized)
        # Convert numpy types to native Python types for JSON serialization
        for entity in results_pre:
            entity['ner_score'] = entity.pop('score')
            entity['ner_score'] = round(float(entity['ner_score']), 4)
        # Align model results to original text offsets
        results = align_results(results_pre, added_spaces_pos, sentence_start_offset)
        return results

    def _process_text(self, text: str) -> list[dict]:
        results_text = []
        line_start_offset = 0  # Track the offset of the start of each line in the file
        for line in text.splitlines():
            doc = self.nlp(line)
            sents = list(doc.sents)
            for sentence in sents:
                results_sent = self._process_sentence(sentence.text, sentence.start_char + line_start_offset)
                results_text.extend(results_sent)
            line_start_offset += len(line) + 1 # account for the '\n' character
        return results_text
    
    def infer(self, texts: list[str]) -> list[list[dict]]:
        return [self._process_text(text) for text in texts]



def ner_inference(texts: list[str], ner_models: list[str], device: str | None = None, agg_strat: str="first") -> list[list[list[dict]]]:
    results = []
    for model_checkpoint in ner_models:
        ner_model = NerModel(model_checkpoint, agg_strat=agg_strat, device=device)
        results_model = ner_model.infer(texts) # Each element in results_model is a list of entities for the corresponding for each text (list[list[dict]])
        results.append(results_model) 
    return results

# TO TEST IN ISOLATION
# import sys

# def main():
#     """
#     this is just for testing purposes.
#     """
#     texts = [
#         "este es un texto de ejemplo.\ncon un paciente procedente de almería aunque nacido en guadalupe, méxico, con mucha tos, mocos, fiebre y la varicela con meningitis.",
#         "otro texto con covid y paracetamol para probar.\ncon más  muchos más síntomas interesantes como edemas."
#     ]
#     model_dir = "/home/bscuser/.cache/huggingface/hub/"

#     model_paths = [
#         "models--BSC-NLP4BIA--bsc-bio-ehr-es-carmen-distemist/snapshots/cd1cbf5dbc13432823f7c1915ef39a350fdd1aa1",
#         "models--BSC-NLP4BIA--bsc-bio-ehr-es-carmen-symptemist/snapshots/a8108ef4d2ee4e4da8ea9f0d8a2fe8d2d2bca367"]
#     results = ner_inference(texts, [model_dir + m_th for m_th in model_paths], agg_strat="first", device="cpu", combined=False)
#     print(results)


# if __name__ == "__main__":
#     sys.exit(main())
