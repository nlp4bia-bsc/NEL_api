"""
ModelResolver
=============
Resolves model identifiers from the registry to local filesystem paths,
downloading from HuggingFace when not already cached.

Resolution order for every transformer model:
  1. target directory already exists  →  cache hit, return immediately
  2. target directory absent          →  download via snapshot_download, then return

Gazetteers are user-supplied and never downloaded. If the TSV is missing,
the resolver raises ModelNotFoundError immediately with an actionable message.
The vectorised DB path is returned regardless of whether the file exists yet —
it will be created at runtime by the NEL component if absent.
"""

from pathlib import Path

from app.config import MODEL_CACHE_DIR, GAZETTEER_CHACHE_DIR
from app.models.registry import GAZ_REGISTRY, NEG_REGISTRY, NEL_REGISTRY, NER_REGISTRY
from app.models.loader import resolve_model_path


class ModelNotFoundError(Exception):
    pass


class ModelResolver:
    def __init__(self, base_dir: str = MODEL_CACHE_DIR):
        self.base_dir = Path(base_dir)
        
    def resolve_ner(self, lang: str, entity_type: str) -> Path:
        """Return local path to the NER model for (lang, entity_type)."""
        try:
            identifier = NER_REGISTRY[lang][entity_type]
        except KeyError:
            raise ModelNotFoundError(
                f"No NER model registered for lang={lang!r}, entity_type={entity_type!r}. "
                f"Add an entry to NER_REGISTRY in app/models/registry.py."
            )
        target = self.base_dir / "ner" / lang / entity_type
        return self._resolve_transformer(identifier, target)

    def resolve_nel(self, lang: str) -> Path:
        """Return local path to the NEL (BiEncoder) model for the given language."""
        try:
            identifier = NEL_REGISTRY[lang]
        except KeyError:
            raise ModelNotFoundError(
                f"No NEL model registered for lang={lang!r}. "
                f"Add an entry to NEL_REGISTRY in app/models/registry.py."
            )
        target = self.base_dir / "nel" / lang
        return self._resolve_transformer(identifier, target)

    def resolve_negation(self, lang: str) -> Path:
        """Return local path to the negation model for the given language."""
        try:
            identifier = NEG_REGISTRY[lang]
        except KeyError:
            raise ModelNotFoundError(
                f"No negation model registered for lang={lang!r}. "
                f"Add an entry to NEG_REGISTRY in app/models/registry.py."
            )
        target = self.base_dir / "negation" / lang
        return self._resolve_transformer(identifier, target)

    def resolve_gazetteer(self, lang: str, entity_type: str) -> tuple[Path, Path]:
        """
        Return (gaz_path, vector_db_path) for (lang, entity_type).

        gaz_path must exist (user-supplied); raises ModelNotFoundError if not.
        vector_db_path is returned as-is — it may not exist yet and will be
        created at runtime by the NEL component.
        """
        try:
            entry = GAZ_REGISTRY[lang][entity_type]
        except KeyError:
            raise ModelNotFoundError(
                f"No gazetteer registered for lang={lang!r}, entity_type={entity_type!r}. "
                f"Add an entry to GAZ_REGISTRY in app/models/registry.py."
            )

        gaz_path = Path(GAZETTEER_CHACHE_DIR) / Path(entry["gaz_path"])
        vector_db_path = Path(GAZETTEER_CHACHE_DIR) / Path(entry["vector_db_path"])

        if not gaz_path.exists():
            raise ModelNotFoundError(
                f"Gazetteer file not found: {gaz_path}\n"
                f"This file must be supplied manually. "
                f"Mount it at the path configured in GAZ_REGISTRY before starting the service.\n"
                f"See README for instructions."
            )

        return gaz_path, vector_db_path


    def _resolve_transformer(self, identifier: str, target: Path) -> Path:
        """
        If target directory already exists, resolve and return the snapshot path.
        Otherwise treat identifier as a HuggingFace repo ID, download, then return
        the snapshot path.

        HuggingFace's snapshot_download stores models under:
        <target>/models--<org>--<name>/snapshots/<hash>/

        The active snapshot hash is recorded in:
        <target>/models--<org>--<name>/refs/main

        We resolve this pointer so callers always receive a path that
        transformers' AutoModel/AutoTokenizer can load directly.
        """
        if target.exists():
            snapshot_path = self._resolve_snapshot_path(target)
            print(f"[ModelResolver] Cache hit: {snapshot_path}")
            return snapshot_path

        print(f"[ModelResolver] Downloading {identifier!r} → {target} ...")
        snapshot_path = resolve_model_path(identifier, target)
        print(f"[ModelResolver] Download complete: {snapshot_path}")
        return snapshot_path

    @staticmethod
    def _resolve_snapshot_path(target: Path) -> Path:
        """
        Given a cache target directory (e.g. .../ner/es/disease), walk into
        the single HuggingFace repo directory it contains and resolve the
        active snapshot via refs/main.

        Raises ModelNotFoundError with an actionable message if the expected
        layout is not found, rather than a cryptic FileNotFoundError downstream.
        """
        repo_dirs = [d for d in target.iterdir() if d.is_dir() and d.name.startswith("models--")]
        if not repo_dirs:
            raise ModelNotFoundError(
                f"Cache directory {target} exists but contains no HuggingFace repo "
                f"subdirectory (expected a directory starting with 'models--'). "
                f"Delete {target} and re-run to trigger a fresh download."
            )
        if len(repo_dirs) > 1:
            raise ModelNotFoundError(
                f"Cache directory {target} contains multiple repo directories: "
                f"{[d.name for d in repo_dirs]}. Expected exactly one. "
                f"Remove unexpected directories manually."
            )

        repo_dir = repo_dirs[0]
        refs_main = repo_dir / "refs" / "main"
        if not refs_main.exists():
            raise ModelNotFoundError(
                f"HuggingFace ref file not found: {refs_main}. "
                f"The download may be incomplete. Delete {target} and re-run."
            )

        snapshot_hash = refs_main.read_text().strip()
        snapshot_path = repo_dir / "snapshots" / snapshot_hash
        if not snapshot_path.exists():
            raise ModelNotFoundError(
                f"Snapshot directory not found: {snapshot_path}. "
                f"The download may be incomplete. Delete {target} and re-run."
            )

        return snapshot_path