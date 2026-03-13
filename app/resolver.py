"""
ModelResolver
=============
Resolves model, gazetteer and vector db paths from the registry using the type of document, self.language and entity if needed

Gazetteers are user-supplied and never downloaded. If the TSV is missing,
the resolver raises ModelNotFoundError immediately with an actionable message.
The vectorised DB path is returned regardless of whether the file exists yet —
it will be created at runtime by the NEL component if absent.
"""

from pathlib import Path
import os
import yaml
from typing import Optional, Any

from app.config import REGISTRY_PATH


class ModelNotFoundError(Exception):
    pass


class LocalResolver:
    def __init__(self, lang: str, entities: list[str]):
        self.cwd = Path(os.getcwd())
        self.reg_path = self.cwd / Path(REGISTRY_PATH)
        self.registry: dict = self._import_registry()
        self.lang = lang
        self.entities = entities
    
    def _import_registry(self) -> dict:
        '''
        Loads the registry.yaml file from the given path, handeling possible errors.
        '''
        try:
            with open(self.reg_path, 'r', encoding = 'utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: The file at {self.reg_path} was not found.")
            return {}
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")
            return {}
    
    def upload_registry(self):
        '''
        Intakes updated registry dict and uploads it to the registry path
        '''
        try:
            with open(self.reg_path, 'w', encoding = 'utf-8') as f:
                yaml.safe_dump(self.registry, stream = f, default_flow_style = False, sort_keys = False)
        except Exception as e:
            print(f"Error updating YAML registry: {e}")

    def resolve_ner(self) -> dict[str, Optional[Path]]:
        try:
            ner_locations = {
                entity_type: Path(pth) if (pth:=self.registry['ner'][self.lang][entity_type]['local_path']) else None
                for entity_type in self.entities
            }

        except KeyError:
            raise ModelNotFoundError(
                f"No NER model registered for language {self.lang!r} for one, some or all entities {self.entities!r}. "
                f"You must add a huggingface id in the registry under that language and entities in app/registry.yaml and run the api again to download the models and save them locally."
            )
        return ner_locations
    
    def resolve_nel(self) -> Optional[Path]:
        try:
            nel_location = Path(pth) if (pth:=self.registry['nel'][self.lang]['local_path']) else None
        except KeyError:
            raise ModelNotFoundError(
                f"No NEL model registered for language {self.lang!r}. "
                f"You must add a huggingface id in the registry under that language in app/registry.yaml and run the api again to download the models and save them locally."
            )
        return nel_location
    
    def resolve_gazetter(self) -> dict[str, Optional[Path]]:
        try:
            gaz_locations = {
                entity_type: Path(pth) if (pth:=self.registry['gazetteers'][self.lang][entity_type]) else None
                for entity_type in self.entities
            }

        except KeyError:
            raise ModelNotFoundError(
                f"No gazetteer registered for language {self.lang!r} for one, some or all entities {self.entities!r}. "
                f"You must add a path to an existing csv or tsv file in the registry under that language and entities in app/registry.yaml."
            )
        return gaz_locations
    
    def resolve_vector_db(self) -> dict[str, Optional[Path]]:
        try:
            vdb_locations = {
                entity_type: Path(pth) if (pth:=self.registry['vectorized_dbs'][self.lang][entity_type]) else None
                for entity_type in self.entities
            }

        except KeyError:
            raise ModelNotFoundError(
                f"No vector databse created for language {self.lang!r} for one, some or all entities {self.entities!r}. "
                f"You must verify the gazetteers and nel models are downloaded and run the api again to generate the vector db."
            )
        return vdb_locations

    def resolve_negation(self) -> Optional[Path]:
        try:
            neg_location = Path(pth) if (pth:=self.registry['ner'][self.lang]['negation']['local_path']) else None
        except KeyError:
            raise ModelNotFoundError(
                f"No NER model registered for language {self.lang!r}. "
                f"You must add a huggingface id in the registry under that language in app/registry.yaml and run the api again to download the models and save them locally."
            )
        return neg_location
    
    def create_paths(self) -> dict[str, Any]:
        ner_paths = {
            ent: self.cwd / 'resources' / 'local_models' / 'ner_models' / self.lang / ent / self.registry['nel'][self.lang].get('repo_id').split('/')[-1]
            for ent in self.entities + ['negation']
        }
        neg_path = self.cwd / 'resources' / 'local_models' / 'neg_model' / self.lang / self.registry["ner"][self.lang]['negation'].get('repo_id').split('/')[-1]
        nel_path = self.cwd / 'resources' / 'local_models' / 'nel_model' / self.lang / self.registry['nel'][self.lang].get('repo_id').split('/')[-1]
        vdb_paths = {ent: self.cwd / 'resources' / 'local_vector_dbs' / self.lang / f"{ent}.pt" for ent in self.entities}
        return {'ner': ner_paths, 'neg': neg_path, 'nel': nel_path, 'vdb': vdb_paths}
