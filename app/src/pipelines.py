from typing import Protocol
from abc import abstractmethod
from pathlib import Path

import pandas as pd

from app.models.resolver import ModelResolver
from app.src.ner import ner_inference
from app.src.nel import lookup_inference, fuzzymatch_inference, biencoder_inference
from app.src.negation import add_negation_uncertainty_attributes
from app.utils.results_postprocessing import join_all_entities
from app.config import LOOKUP_PATH


class AnnotationPipeline(Protocol):
    @abstractmethod
    def predict(self, texts: list[str]) -> list[list[dict]]:
        """
        Run the full annotation pipeline over a batch of input texts.

        Parameters
        ----------
        texts : list[str]
            A list of raw input strings. Each element is processed independently
            and may contain zero or more detectable entity mentions.

        Returns
        -------
        list[list[dict]]
            A list with the same length as `texts`. Each element corresponds to
            one input text and contains a list of annotation dictionaries.

            Each annotation dictionary represents a single detected entity
            mention enriched with normalization and contextual attributes,
            with the following schema:

            {
                "start": int,
                    Character-level start offset of the entity span
                    (inclusive, 0-based index).

                "end": int,
                    Character-level end offset of the entity span
                    (exclusive).

                "span": str,
                    Exact substring of the original text corresponding
                    to the detected entity mention.

                "ner_class": str,
                    Predicted named entity class/category.

                "ner_score": float,
                    Confidence score assigned by the NER component.

                "code": str,
                    Normalized identifier assigned by the entity
                    linking / normalization component.

                "term": str,
                    Canonical term associated with the predicted code.

                "nel_score": float,
                    Confidence score assigned by the normalization step.

                "is_negated": bool,
                    Whether the entity mention is predicted to be
                    negated in context.

                "negation_score": float,
                    Confidence score of the negation prediction.

                "is_uncertain": bool,
                    Whether the entity mention is predicted to be
                    expressed with uncertainty/speculation.

                "uncertainty_score": float,
                    Confidence score of the uncertainty prediction.
            }

        Notes
        -----
        - The outer list preserves input order.
        - If no entities are detected in a text, the corresponding element
          will be an empty list.
        - Character offsets must refer to the original, unmodified input text.
        """
        pass


class LookupPipeline(AnnotationPipeline):
    """Direct text → code lookup. No NER step needed."""
    def __init__(self, lang: str, entities: list[str]):
                
        # list of gath paths
        resolver = ModelResolver()
        self.gazeteer_pths = [resolver.resolve_gazetteer(lang, e)[0] for e in entities] 

    def predict(self, texts: list[str]) -> list[list[dict]]:
        
        # obtain results
        inference_results = lookup_inference(texts, self.gazeteer_pths, self.case_sensitive)
        
        # return flattened results: list[list[list[dict]]] -> list[list[dict]]
        return join_all_entities(inference_results)
    
    
class FuzzyMatchPipeline(AnnotationPipeline):
    def __init__(self, lang: str, entities: list[str], method = "jaro_winkler", threshold = 0.7, agg_strat = "first", device = None):
        self.device = device

        # method parameters
        self.method = method
        self.threshold = threshold
        
        resolver = ModelResolver()

        # NER paths: one ner per entity
        self.ner_pths = [resolver.resolve_ner(lang, e) for e in entities]
        self.agg_strat = agg_strat
        
        # list of gath paths
        self.gaz_pths = [resolver.resolve_gazetteer(lang, e)[0] for e in entities]    
        
        self.device = device     
        
    def predict(self, texts: list[str]) -> list[list[dict]]:
        
        # run NER per entity
        ner_results = ner_inference(texts, self.ner_pths, agg_strat = self.agg_strat, device = self.device)
        
        # run fuzzy match on extracted entities
        fuzzy_result = fuzzymatch_inference(ner_results, self.gaz_pths, self.method, self.threshold)
        
        # return flattened results: list[list[list[dict]]] -> list[list[dict]]
        return join_all_entities(fuzzy_result)


class BiencoderPipeline(AnnotationPipeline):
    """
    Full pipeline: NER → NEL (dense retrieval) → Negation.

    Parameters
    ----------
    lang : str
        Language code, e.g. "es". All texts in a request are assumed to share
        this language.
    entities : list[str]
        Entity types to detect, e.g. ["disease", "symptoms"].
        Each type must have a corresponding NER entry and gazetteer in the
        registry. A single NEL model and a single negation model are shared
        across all entity types for the language.
    agg_strat : str
        NER aggregation strategy passed to ner_inference. Default "first".
    device : str or None
        Torch device string, e.g. "cuda:0". None = auto-select.
    """

    def __init__(
        self,
        lang: str,
        entities: list[str],
        agg_strat: str = "first",
        device=None,
    ):
        resolver = ModelResolver()

        # One NER model per entity type
        self.ner_paths = [resolver.resolve_ner(lang, e) for e in entities]

        # One NEL (BiEncoder) model shared across all entity types for this language
        self.nel_path = resolver.resolve_nel(lang)

        # Per-entity gazetteers: list of (gaz_path, vector_db_path)
        self.gaz_and_vdb = [resolver.resolve_gazetteer(lang, e) for e in entities]

        # Single negation model for this language, decoupled from the NER list
        self.neg_path = resolver.resolve_negation(lang)

        self.agg_strat = agg_strat
        self.device = device

    def predict(self, texts: list[str]) -> list[list[dict]]:
        # Entity NER — one model per entity type
        ner_results = ner_inference(
            texts, self.ner_paths, agg_strat=self.agg_strat, device=self.device
        )
        # Negation — separate pass with its own model
        # NOTE: if negation.py has a dedicated inference function with a different
        # signature than ner_inference, replace this call with that function.
        neg_results = ner_inference(
            texts, [self.neg_path], agg_strat=self.agg_strat, device=self.device
        )
        # NEL normalisation — single BiEncoder, per-entity gazetteers
        norm_results = biencoder_inference(
            ner_results, self.nel_path, self.gaz_and_vdb, device=self.device
        )
        norm_results = join_all_entities(norm_results)
        final_results = add_negation_uncertainty_attributes(norm_results, neg_results[0])
        return final_results
