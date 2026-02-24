import sys, os
import pandas as pd
import torch
import copy

from utils.model_utils import DenseRetriever
from sentence_transformers import SentenceTransformer
from utils.results_postprocessing import join_all_entities

class NelModel:
    def __init__(self, gaz_pth: str, model_pth: str, vector_db_pth: str, device: str | None = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.st_model = SentenceTransformer(model_pth).to(self.device)
        self.gazeteer = pd.read_csv(gaz_pth, sep='\t')
        self._load_vector_db(vector_db_pth)
        self.biencoder = DenseRetriever(
            gazeteer_df=self.gazeteer, 
            vector_db=self.vector_db, 
            model_or_path=self.st_model
        )
    
    def _load_vector_db(self, vector_db_path: str):
        if os.path.exists(vector_db_path):
            print("Loading vector database from file...")
            print(self.device)
            self.vector_db = torch.load(vector_db_path, map_location=self.device)
        else:
            print("Vector database not found. Computing vector database...")
            terms = self.gazeteer['terms'].unique()
            self.vector_db = self.st_model.encode(
                terms,
                show_progress_bar=True, 
                convert_to_tensor=True,
                batch_size=4096,
                device=self.device.type  # make sure encoding runs on the same device
            )
            torch.save(self.vector_db, vector_db_path)
            print(f"Vector database saved at {vector_db_path}")

    def run_nel_inference(self, input_mentions: list, k: int=10) -> pd.DataFrame:
        """
        Returns a dataframe where the index is the span and the and the vaues are the code, term, and simmilarity. It can be accessed through df.loc['covid'] --> 1119302008 / 'COVID-19 agudo' / 0.7942
        """
        mentions = list(set(input_mentions)) # filter duplicates

        candidates = self.biencoder.retrieve_top_k(
            mentions, 
            k=k, 
            input_format="text",
            return_documents=True
        )

        candidates_df = pd.DataFrame(candidates)
        # convert 1-element-list values to single values
        candidates_df = candidates_df.explode(['codes', 'terms', 'similarity'])
        candidates_df = candidates_df.rename(columns={'codes': 'code', 'terms': 'term'}) # singular 
        candidates_df["similarity"] = candidates_df["similarity"].apply(
            lambda sim: round(sim, 4)
        )
        return candidates_df.set_index('mention')


def nel_inference(ner_results: list[list[list[dict]]], model_paths: list[tuple[str, str, str]], combined: bool=True, device: str | None = None):
    """
    ner_results = [//result level
        [// entity type level
            [// doc level
                {'start': 136, 'end': 159, 'ner_score': 0.9999, 'span': 'varicela con meningitis', 'ner_class': 'ENFERMEDAD'}
            ], 
            [
                {'start': 15, 'end': 20, 'ner_score': 0.9999, 'span': 'covid', 'ner_class': 'ENFERMEDAD'}
            ]
        ], (...)
    ]

    path_list = [
        "(<gazetteer_path>, <model_path>, <vector_db_path> | None)"
    ]

    combined is a bool, fool
    """

    assert len(ner_results) == len(model_paths)

    nerl_results = ner_results.copy()
    for ent_type_idx, (ent_type_mentions, (gaz_pth, nel_model_pth, vector_db_pth)) in enumerate(zip(ner_results, model_paths)): # will iterate over all entity types (both in ner results and nel models)
        mentions = [mention_dict['span'] for mention_doc in ent_type_mentions for mention_dict in mention_doc]
        if len(mentions) == 0:
            continue # no mentions for that entity type

        nel_model = NelModel(gaz_pth=gaz_pth,
                             model_pth=nel_model_pth,
                             vector_db_pth=vector_db_pth,
                             device=device)
        
        output = nel_model.run_nel_inference(
            input_mentions=mentions,
            k=1 # we only want the top decision
        )

        for mention_doc in nerl_results[ent_type_idx]: # same as mentions list comprehension, exploit the FACT that it has same order
            for mention_dict in mention_doc:
                mention_dict["code"], mention_dict["term"], mention_dict["nel_score"] = output.loc[mention_dict["span"]]
                
    if combined:
        nerl_results = join_all_entities(nerl_results)

    return nerl_results



import sys

def main():
    """
    this is just for testing purposes.
    """
    texts = [
        "este es un texto de ejemplo.\ncon un paciente procedente de almería aunque nacido en guadalupe, méxico, con mucha tos, mocos, fiebre y la varicela con meningitis.",
        "otro texto con covid y paracetamol para probar.\ncon más  muchos más síntomas interesantes como edemas."
    ]

    path_list = [
        "(<gazetteer_path>, <model_path>, <vector_db_path> | None)"
    ]
    # model_dir = "/home/bscuser/.cache/huggingface/hub/"

    # ner_model_paths = [
    #     "models--BSC-NLP4BIA--bsc-bio-ehr-es-carmen-distemist/snapshots/cd1cbf5dbc13432823f7c1915ef39a350fdd1aa1",
    #     "models--BSC-NLP4BIA--bsc-bio-ehr-es-carmen-symptemist/snapshots/a8108ef4d2ee4e4da8ea9f0d8a2fe8d2d2bca367"]

    ner_results = ner_inference(texts, [model_dir + m_th for m_th in ner_model_paths], agg_strat="first", device="cpu", combined=False)
    nel_results = nel_inference(ner_results, combined_True)
    print(nel_results)


if __name__ == "__main__":
    sys.exit(main())
