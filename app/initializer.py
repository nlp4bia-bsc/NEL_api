import csv
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import torch
import time
import gc

from app.utils.download_model import create_vector_db
from app.resolver import LocalResolver


class ResourceDownloader:
    """
    Handles validation and downloading of all model assets (NER, NEL, gazetteers,
    and vector databases) required by the pipeline for a given language and set of entities.

    Intended to be run once at setup time. After downloading, local paths are
    persisted back to the registry so that LocalResolver can find them at runtime.
    """

    def __init__(self, lang: str, entities: list[str], negation: bool=False, device: str='cuda'):
        self.lang = lang
        self.entities = entities
        self.negation = negation
        self.device = device

        self.resolver = LocalResolver(self.lang)
        self.registry = self.resolver.registry

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def check_gazetteers(self) -> None:
        """
        Checks that each entity is present in the registry and that its
        associated gazetteer file exists and contains the required columns.
        """
        required_cols = {"term", "code"}

        for entity in self.entities:
            gaz_path = self.resolver.get_gaz_path(entity)
            
            if gaz_path.suffix == '.csv':
                tsv_path = gaz_path.with_suffix('.tsv')
                with open(gaz_path, newline='') as f_in, open(tsv_path, 'w', newline='') as f_out:
                    f_out.write(f_in.read().replace(',', '\t'))
                self.registry['gazetteers'][self.lang][entity] = str(tsv_path)  # update registry in place
                gaz_path = tsv_path

            with open(gaz_path, newline="") as f:
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
        entities2download = self.entities + ["negation"] if self.negation else self.entities
        for entity in entities2download:
            ner_local_path, repo_id = self.resolver.get_ner_path(entity)

            if repo_id: # if repo id is returned (not None), it means it has to be downloaded
                ner_local_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading NER model for {self.lang!r} / {entity!r} from {repo_id!r}...")
                snapshot_download(
                    repo_id = repo_id,
                    local_dir = ner_local_path
                )
                self.registry["ner"][self.lang][entity]["local_path"] = str(ner_local_path)
            else: # repo_id is none, means the model is already downloaded
                print(f"Model for {self.lang!r} / {entity!r} already downloaded in {str(ner_local_path)!r}")

    def download_nel(self) -> None:
        """
        Checks whether the NEL model is already present locally; downloads it
        if not.
        """
        nel_local_path, repo_id = self.resolver.get_nel_path()
        if repo_id: # if repo id is returned (not None), it means it has to be downloaded
            nel_local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading NEL model for {self.lang!r} from {repo_id!r}...")
            snapshot_download(
                repo_id = repo_id,
                local_dir = nel_local_path
            )
            self.registry["nel"][self.lang]["local_path"] = str(nel_local_path)
        else: # repo_id is none, means the model is already downloaded
            print(f"Model for {self.lang!r} already downloaded in {str(nel_local_path)!r}")

    def _get_gaz_terms(self, entity: str) -> list[str]:
        gaz_pth = self.resolver.get_gaz_path(entity)
        gaz_df = pd.read_csv(gaz_pth, sep="\t")
        terms_array = gaz_df["term"].unique()
        del gaz_df 
        gc.collect()
        return list(terms_array)

    def download_vector_db(self) -> None:
        """
        Creates a vector database for each entity whose DB is not already
        registered. Uses the NEL sentence-transformer model to encode gazetteer
        terms.
        """
        # check if you actually have to create any of them to avoid unnecessarily downloading the nel model
        vector_db_pths = {
            ent: res[0]
            for ent in self.entities
            if not (res := self.resolver.get_vector_db_path(ent))[1]
        }

        if len(vector_db_pths) == 0: # no vector db needed to compute, hence no need to load nel model
            print(f"All vector dbs for {self.lang!r} and {self.entities!r} have already been downloaded!")
            return
        
        nel_local_path, _ = self.resolver.get_nel_path()
        if not nel_local_path:
            raise ValueError("The NEL model has not been created succesfully.")
        nel_model = SentenceTransformer(str(nel_local_path))
        
        for entity, vector_db_pth in vector_db_pths.items():
            vector_db_pth.parent.mkdir(parents=True, exist_ok=True)
            gaz_terms = self._get_gaz_terms(entity)
            create_vector_db(gaz_terms, nel_model, vector_db_pth, self.device)
            del gaz_terms
            # help the OS finalize file handles
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
            self.registry["vectorized_dbs"][self.lang][entity] = str(vector_db_pth)
            self.resolver.upload_registry() # a very expensive computation. Save in case future loads are not correctly performed

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
        self.resolver.upload_registry()

        print("Downloading NER models...")
        self.download_ner()
        self.resolver.upload_registry()

        print("Downloading NEL model...")
        self.download_nel()
        self.resolver.upload_registry()

        print("Building vector databases...")
        self.download_vector_db()
        self.resolver.upload_registry()

        print("Setup complete.")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main(lang: str, entities: list[str], negation: bool) -> None:
    ResourceDownloader(lang, entities, negation).run()