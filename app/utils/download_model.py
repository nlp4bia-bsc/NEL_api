from pathlib import Path
import pandas as pd 
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import torch
import numpy as np
import gc
from tqdm import tqdm

def HF_download_model(repo_id: str, path: Path) -> str:
    '''
    Given a model repo_id and path name, it downlaods the model in model cache and returns the new local path
    '''
    snapshot_download(
        repo_id = repo_id,
        local_dir = path
    )
    
    return str(path)


def _create_vector_db(gazetteer: pd.DataFrame, nel_model: SentenceTransformer, vector_db_path: Path, device: str, chunk_size: int=10000): # Smaller chunk size
    terms = gazetteer["term"].tolist()
    num_terms = len(terms)
    embedding_dim = 768 

    fp = np.memmap(vector_db_path, dtype=np.float32, mode='w+', shape=(num_terms, embedding_dim))

    print("Computing vector database...")
    for i in tqdm(range(0, num_terms, chunk_size)):
        end_idx = min(i + chunk_size, num_terms)
        chunk = terms[i:end_idx]

        with torch.no_grad(): # Ensure no gradients are stored (saves massive memory)
            embeddings = nel_model.encode(
                chunk,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1024,
                device=device
            )
        if "cuda" in str(device):
            torch.cuda.empty_cache()

        fp[i:end_idx, :] = embeddings
        fp.flush()
        del chunk, embeddings
        gc.collect()

    del fp 


def load_as_torch_tensor(vector_db_path: Path, gazz_terms: int, embedding_dim: int = 768, device: str='cuda') -> torch.Tensor:
    nmap = np.memmap(vector_db_path, dtype='float32', mode='r', shape=(gazz_terms, embedding_dim))
    torch_db = torch.from_numpy(nmap)
    return torch_db.to(device=device)