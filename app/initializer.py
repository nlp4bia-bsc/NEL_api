import yaml
import argparse
from pathlib import Path

from app.config import REGISTRY_PATH
from app.utils.download_model import HF_download_model

def import_registry(path) -> dict: 
    '''
    Loads the registry.yaml file from the given path, handeling possible errors.
    '''
    
    try:
        with open(path, 'r', encoding = 'utf-8') as file:
            # using safe_load to ensure correct reads
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML: {exc}")
        return {}
    

def check_gazzetteers(gazzetters: dict, entities: list[str]) -> str:
    '''
    Checks that the entities are present in the registry, and that the associated paths are correct.
    '''
    
    # errors = []
    
    for entity in entities:
        
        # check if entity is present
        if entity not in gazzetters:
            raise ValueError(f"Entity {entity!r} was not found in registry.")
            # errors.append(f"Entity {entity!r} was not found in registry.")

        
        # check path of that entity
        path = Path(gazzetters[entity])
        if not path.is_file():
            raise FileNotFoundError(f"File for {entity!r} not found at {path!r}")
            # errors.append(f"File for {entity!r} not found at {path!r}")
    
    # # present any errors raised
    # if errors:
    #     error_msg = "\n- ".join(["Registry check failed:"] + errors)
    #     raise RuntimeError(error_msg)
    
    
def download_ner(ners: dict, entities: list[str]) -> None:
    '''
    Checks the ner models in registry and downloads those that are not already present in local.
    '''
    
    for entity in entities:
        
        # check if the entity has a ner model
        if entity not in ners: raise ValueError(f"Entity {entity!r} has no associated NER model in the registry.")
        
        # get repo id: mandatory, even for personal use
        repo_id = ners[entity].get('repo_id')
        if not repo_id: 
            raise ValueError(f"The entity {entity!r} has no associated HuggingFace.")
        
        # get local path
        local_path = ners[entity].get('local_path')
        
        # check if there's a local path for the model
        if not local_path: 
            
            #download model
            local_path = HF_download_model(repo_id, path = f"ner_models/{entity}")
            
            # save local directory to registry
            ners[entity]['local_path'] = local_path
        
        # check if local path exists    
        else:   
            local_path = Path(local_path)
            if not local_path.is_dir():
                raise ValueError(f"The NER local path provided for {entity!r} does not exist: {local_path!r}") 
               

def download_nel(nel_model: dict):
    '''
    Checks whether the NEL model is laready present locally, if not, downloads it in correct location.    
    '''
    
    repo_id = nel_model.get('repo_id')
    if not repo_id: 
        raise ValueError(f"The nel model has no associated HuggingFace id.")
    
    # get local path
    local_path = nel_model.get('local_path')
    
    if not local_path:
        
        model_name = repo_id.split('/')[-1] # model name for dir
        
        # downlaod model
        local_path = HF_download_model(repo_id, path = f"nel_models/{model_name}")
        
        # save local directory to registry
        nel_model['local_path'] = local_path
    
    # check if local path exists
    else:   
        local_path = Path(local_path)
        if not local_path.is_dir(): 
            raise ValueError(f"The local path provided does not exist: {local_path!r}") 
        
        
def upload_registry(registry_path: Path, registry: dict):
    '''
    Intakes updated registry dict and uploads it to the registry path
    '''
    
    try:
        with open(registry_path, 'w', encoding = 'utf-8') as f:
            yaml.safe_dump(registry_path, stream = f, default_flow_style = False, sort_keys = False)
    except Exception as e:
        print(f"Error updating YAML registry: {e}")


def main(lang: str, entities: list[str]):
    registry = import_registry(REGISTRY_PATH)
    check_gazzetteers(registry['gazetteers'][lang], entities) 
    download_ner(registry['ner'][lang], entities) 
    download_nel(registry['nel'][lang])
    create_vector_db(registry['gazetteers'], registry['nel'], entities)
    upload_registry(REGISTRY_PATH, registry)


if __name__ == '__main__':
    main()