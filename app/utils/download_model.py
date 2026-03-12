import config

from pathlib import Path
from huggingface_hub import snapshot_download

def HF_download_model(repo_id: str, path: str):
    '''
    Given a model repo_id and path name, it downlaods the model in model cache and returns the new local path
    '''

    cache_path = Path(config.MODEL_CACHE_DIR)
    local_path = cache_path / path
    
    snapshot_download(
        repo_id = repo_id,
        local_dir = local_path
    )
    
    return str(local_path)