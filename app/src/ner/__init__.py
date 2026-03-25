from .encoder_inference_v1 import ner_inference_v1
from .encoder_inference_v2 import ner_inference_v2
from pathlib import Path
from typing import Optional


"""
Regardless of the version, the inference function must take a list of texts to infer on and a list of models to 
{
    "start": int,
    "end": int,
    "ner_score": float,
    "span": str,
    "ner_class": Literal["ENFERMEDAD", "PROCEDIMIENTO", "SINTOMA", "NEGACIÓN"],
    "code": str,
    "term": str,
    "nel_score": float
}
"""


def encoder_inference(
    # necessary inputs
    texts: list[str],
    ner_models: list[Path],
    version: int = 2,
    # v1 & v2 shared
    device: str = 'cuda',
    agg_strat: Optional[str] = None,
    # v1-only
    lang: str = "es",
    # v2-only
    filenames: Optional[list[str]] = None,
    batch_size: int = 16,
    merge_entities: bool = True,
    score_mode: str = "mean",
) -> list[list[list[dict]]]:
    """
    Unified entry point for NER inference.

    Args:
        texts:          Input texts to annotate.
        ner_models:     Paths to the model checkpoints.
        version:        Which inference backend to use (1 or 2). Defaults to 2.
        device:         Torch device string, e.g. "cuda".
        agg_strat:      Aggregation strategy. Defaults to "first" for v1, "simple" for v2.
        lang:           (v1 only) Language code. Defaults to "es".
        filenames:      (v2 only) Optional filenames aligned with texts.
        batch_size:     (v2 only) Inference batch size. Defaults to 16.
        merge_entities: (v2 only) Whether to merge adjacent entities. Defaults to True.
        score_mode:     (v2 only) How to aggregate token scores. Defaults to "mean".

    Returns:
        Nested list of entity dicts: [text][model][entity].
    """
    if version == 1:
        return ner_inference_v1(
            texts=texts,
            ner_models=ner_models,
            device=device,
            agg_strat=agg_strat if agg_strat is not None else "first",
            lang=lang,
        )
    elif version == 2:
        return ner_inference_v2(
            texts=texts,
            ner_models=ner_models,
            device=device,
            agg_strat=agg_strat if agg_strat is not None else "simple",
            filenames=filenames,
            batch_size=batch_size,
            merge_entities=merge_entities,
            score_mode=score_mode,
        )
    else:
        raise ValueError(f"Unknown version {version!r}. Expected 1 or 2.")