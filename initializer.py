import csv
import yaml
import torch
import argparse
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import gc
from tqdm import tqdm

from app.config import REGISTRY_PATH
from app.utils.download_model import HF_download_model, _create_vector_db
from app.utils.model_utils import DenseRetriever
from app.resolver import LocalResolver


class ResourceDownloader:
    """
    Handles validation and downloading of all model assets (NER, NEL, gazetteers,
    and vector databases) required by the pipeline for a given language and set of entities.

    Intended to be run once at setup time. After downloading, local paths are
    persisted back to the registry so that LocalResolver can find them at runtime.
    """

    def __init__(self, lang: str, entities: list[str], negation: bool=True, device: str='cuda'):
        self.lang = lang
        self.entities = entities
        self.negation = negation
        self.device = device

        self.resolver = LocalResolver(lang, entities)
        self.registry = self.resolver.registry
        self.new_paths = self.resolver.create_paths()

    # ------------------------------------------------------------------
    # Registry I/O
    # ------------------------------------------------------------------

    def _upload_registry(self) -> None:
        self.resolver.upload_registry()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def check_gazetteers(self) -> None:
        """
        Checks that each entity is present in the registry and that its
        associated gazetteer file exists and contains the required columns.
        """
        gazetteers = self.registry["gazetteers"][self.lang]
        required_cols = {"term", "code"}

        for entity in self.entities:
            if entity not in gazetteers:
                raise ValueError(f"Entity {entity!r} was not found in registry.")

            path = Path(gazetteers[entity])
            if not path.is_file():
                raise FileNotFoundError(
                    f"Gazetteer file for {entity!r} not found at {path!r}."
                )
            
            if path.suffix == '.csv':
                tsv_path = path.with_suffix('.tsv')
                with open(path, newline='') as f_in, open(tsv_path, 'w', newline='') as f_out:
                    f_out.write(f_in.read().replace(',', '\t'))
                path = tsv_path
                gazetteers[entity] = str(tsv_path)  # update registry in place

            with open(path, newline="") as f:
                headers = set(next(csv.reader(f, delimiter="\t")))

            missing = required_cols - headers
            if missing:
                raise ValueError(
                    f"Gazetteer for {entity!r} is missing columns: {missing}."
                )

    # ------------------------------------------------------------------
    # Downloading
    # ------------------------------------------------------------------

    def download_ner(self) -> None:
        """
        Checks NER model entries in the registry and downloads any that are
        not already present locally.
        """
        ner_hf_repos = self.registry["ner"][self.lang]
        ner_local_paths = self.resolver.resolve_ner()
        for entity in self.entities:
            if entity not in ner_hf_repos:
                raise ValueError(
                    f"Entity {entity!r} has no associated NER model in the registry."
                )

            ner_repo_id = ner_hf_repos[entity].get("repo_id")
            if not ner_repo_id:
                raise ValueError(
                    f"Entity {entity!r} has no associated HuggingFace repo id."
                )

            ner_local_path = ner_local_paths[entity]

            if not ner_local_path:
                new_path = self.new_paths['ner'][entity]
                new_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading NER model for {ner_repo_id!r}...")
                str_new_path = HF_download_model(ner_repo_id, path=new_path)
                self.registry["ner"][self.lang][entity]["local_path"] = str_new_path
            else:
                if not ner_local_path.is_dir():
                    raise ValueError(
                        f"NER local path for {entity!r} does not exist: {ner_local_path!r}."
                    )
    def download_neg(self) -> None:
        """
        Checks negation model entries in the registry and downloads if
        not already present locally.
        """
        neg_hf_repo = self.registry["ner"][self.lang]["negation"].get("repo_id")
        neg_local_path = self.resolver.resolve_negation()
        if not neg_hf_repo:
            raise ValueError("The NEGATION model has no associated HuggingFace repo id.")

        if not neg_local_path:
            new_path = self.new_paths['neg']
            new_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading NEG model {neg_hf_repo!r}...")
            str_new_path = HF_download_model(neg_hf_repo, path=new_path)
            self.registry["ner"][self.lang]["negation"]["local_path"] = str_new_path
        else:
            if not Path(neg_local_path).is_dir():
                raise ValueError(
                    f"NEL local path does not exist: {neg_local_path!r}."
                )

    def download_nel(self) -> None:
        """
        Checks whether the NEL model is already present locally; downloads it
        if not.
        """
        nel_hf_repo = self.registry["nel"][self.lang].get("repo_id")
        nel_local_path = self.resolver.resolve_nel()
        if not nel_hf_repo:
            raise ValueError("The NEL model has no associated HuggingFace repo id.")

        if not nel_local_path:
            new_path = self.new_paths['nel']
            new_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading NEL model {nel_hf_repo!r}...")
            str_new_path = HF_download_model(nel_hf_repo, path=new_path)
            self.registry["nel"][self.lang]["local_path"] = str_new_path
        else:
            if not Path(nel_local_path).is_dir():
                raise ValueError(
                    f"NEL local path does not exist: {nel_local_path!r}."
                )

    def download_vector_db(self) -> None:
        """
        Creates a vector database for each entity whose DB is not already
        registered. Uses the NEL sentence-transformer model to encode gazetteer
        terms.
        """
        gaz_registry = self.registry["gazetteers"][self.lang]
        nel_local_path = self.resolver.resolve_nel()
        vdb_registry = self.resolver.resolve_vector_db()

        if not nel_local_path:
            raise ValueError("The NEL model has not been created succesfully.")
        
        nel_model = SentenceTransformer(str(nel_local_path))

        for entity in self.entities:
            if vdb_registry.get(entity):
                print(
                    f"Vector database for {entity!r} already present at "
                    f"{vdb_registry[entity]!r}."
                )
                continue

            vector_db_pth: Path = self.new_paths["vdb"][entity]
            vector_db_pth.parent.mkdir(parents=True, exist_ok=True)

            print(f"Building vector database for {entity!r}...")
            ent_gaz = pd.read_csv(gaz_registry[entity], sep="\t")
            ent_gaz = ent_gaz.drop_duplicates(subset=["term"])
            _create_vector_db(ent_gaz, nel_model, vector_db_pth, self.device)

            self.registry["vectorized_dbs"][self.lang][entity] = str(vector_db_pth)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Runs the full setup pipeline: validate gazetteers, download NER models,
        download NEL model, and build vector databases. The registry is saved
        after each download step so partial progress is not lost on failure.
        """
        print("Checking gazetteers...")
        self.check_gazetteers()

        print("Downloading NER models...")
        self.download_ner()
        self._upload_registry()

        print("Downloading NEGATION model...")
        self.download_neg()
        self._upload_registry()

        print("Downloading NEL model...")
        self.download_nel()
        self._upload_registry()

        print("Building vector databases...")
        self.download_vector_db()
        self._upload_registry()

        print("Setup complete.")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main(lang: str, entities: list[str], negation: bool) -> None:
    ResourceDownloader(lang, entities, negation).run()