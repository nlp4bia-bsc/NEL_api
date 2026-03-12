"""
Model Loader
============
Single responsibility: given a HuggingFace repo ID or a local path,
return a guaranteed-valid local directory path.

No model objects are instantiated here. The actual AutoModel / tokenizer
loading stays in ner.py, nel.py, negation.py as before.
"""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def resolve_model_path(repo_id_or_path: str, cache_dir: str | Path) -> Path:
    """
    Resolve a model reference to a local directory path.

    If ``repo_id_or_path`` is an existing local path it is returned as-is.
    Otherwise it is treated as a HuggingFace repo ID and downloaded into
    ``cache_dir`` (skipped if already cached).

    Parameters
    ----------
    repo_id_or_path : str
        A HuggingFace repo ID (e.g. ``"org/model-name"``) or an absolute /
        relative path to an existing local model directory.
    cache_dir : str or Path
        Root directory under which HF snapshots are stored.
        Recommended: ``/app/local_models`` (volume-mounted in Docker).

    Returns
    -------
    str
        Absolute path to the local model directory.
   """
    local = Path(repo_id_or_path)

    # --- local path branch ---------------------------------------------------
    # Treat as local if it's absolute, starts with ./ or ../, or the path exists.
    if local.exists() and (local / "config.json").exists():
        logger.debug("Found valid local model at: %s", local.resolve())
        return local.resolve()

    # --- HuggingFace branch --------------------------------------------------
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Resolving model %r from HuggingFace (cache: %s)", repo_id_or_path, cache_dir)
    local_path = snapshot_download(
        repo_id=repo_id_or_path,
        cache_dir=str(cache_dir),
        local_files_only=False,
    )
    logger.info("Model ready at: %s", local_path)
    return Path(local_path)
