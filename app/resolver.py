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
from typing import Optional

from app.config import REGISTRY_PATH


class ModelNotFoundError(Exception):
    pass


class LocalResolver:
    def __init__(self, lang: str):
        self.cwd = Path(os.getcwd())
        self.base_pth =  self.cwd / "app" / "resources"
        self.reg_path = self.cwd / Path(REGISTRY_PATH)
        self.registry: dict = self._import_registry()
        self.lang = lang
    
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

    def get_ner_path(self, entity: str) -> tuple[Path, Optional[str]]:
        try:
            pth = self.registry['ner'][self.lang][entity]['local_path']
        except KeyError:
            raise ModelNotFoundError(f"No NER model registered for {self.lang!r} / {entity!r}.")
        if pth is None:
            try:
                repo_id = self.registry['ner'][self.lang][entity]['repo_id']
            except:
                raise ModelNotFoundError(f"No repo id provided for the NER model responsible for {self.lang!r} / {entity!r}.")
            pth = (
                self.base_pth / 'local_models' / 'ner_models'
                / self.lang / entity / repo_id.split('/')[-1]
            )
            print(f"NER model for {self.lang!r} / {entity!r} has not been downloaded yet, downloading it now from hf: {repo_id!r} into {pth!r}.")
            return Path(pth), repo_id
        
        elif not Path(pth).exists(): # verify it actually exists
             raise FileNotFoundError(
                    f"Ner Model for {self.lang!r} {entity!r} not found at {pth!r} specified in the registry."
                )
        # model already downloaded          
        return Path(pth), None
    
    def get_nel_path(self) -> tuple[Path, Optional[str]]:
        try:
            pth = self.registry['nel'][self.lang]['local_path']
        except KeyError:
            raise ModelNotFoundError(f"No NEL model registered for {self.lang!r}.")
        if pth is None:
            try:
                repo_id = self.registry['nel'][self.lang]['repo_id']
            except:
                raise ModelNotFoundError(f"No repo id provided for the NEL model responsible for {self.lang!r}.")
            pth = Path(
                self.base_pth / 'local_models' / 'nel_models'
                / self.lang / repo_id.split('/')[-1]
            )
            print(f"NEL model for {self.lang!r} has not been downloaded yet, downloading it now from hf: {repo_id!r} into {pth!r}.")
            return Path(pth), repo_id
        
        elif not Path(pth).exists(): # verify it actually exists
             raise FileNotFoundError(
                    f"Ner Model for {self.lang!r} not found at {pth!r} specified in the registry."
                )
        # model already downloaded
        return Path(pth), None
    
    def get_gaz_path(self, entity: str) -> Path:
        try:
            pth = Path(self.registry['gazetteers'][self.lang][entity])

        except KeyError:
            raise ModelNotFoundError(
                f"No gazetteer registered for language {self.lang!r} for {entity!r}. "
                f"You must add an absolute path to an existing csv or tsv file in the registry under that language and entity in app/registry.yaml."
            )
        
        if not Path(pth).exists():
            raise FileNotFoundError(
                    f"Gazetteer file for {self.lang!r} {entity!r} not found at {pth!r} specified in the registry."
                )
        # gazetteer is correctly referenced
        return pth
    
    def get_vector_db_path(self, entity: str) -> tuple[Path, bool]:
        downloaded = True
        try:
            pth = self.registry['vectorized_dbs'][self.lang][entity]
        except KeyError:
            raise ModelNotFoundError(f"No vectorized db created for {entity!r} under language {self.lang!r} .")
        if pth is None:
            pth = Path(
                self.base_pth / 'vectorized_dbs' / self.lang / f"{entity}.pt"
            )
            downloaded = False
            print(f"No vectorized db created for {entity!r} under language {self.lang!r}. Creating one using the default NEL model / gazetteer for that language and entity combination")
        elif not Path(pth).exists():
            raise FileNotFoundError(
                    f"Vector DB for {self.lang!r} {entity!r} not found at {pth!r} specified in the registry."
                )
        return Path(pth), downloaded