import sys, os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from app.utils.model_utils import DenseRetriever

class NelModel:
    def __init__(self, gaz_pth: str, model_pth: str, vector_db_pth: str, device: str | None = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

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


def nel_inference(ner_results: list[list[list[dict]]], model_paths: list[tuple[str, str, str]], device: str | None = None) -> list[list[list[dict]]]:
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

    returns the same ner_results list of list of list of dict with extra keys for the normalized codes and the simmilarity to the original concept
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
                
    return nerl_results
