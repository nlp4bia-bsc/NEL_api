from typing import Protocol
from abc import abstractmethod
import pandas as pd

from app.config import NER_PATHS, NEL_PATHS, LOOKUP_PATH
from app.src.ner import ner_inference
from app.src.nel import lookup_inference, biencoder_inference
from app.src.negation import add_negation_uncertainty_attributes
from app.utils.results_postprocessing import join_all_entities

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
    def __init__(self, case_sensitive = False):
        self.case_sensitive = case_sensitive
        self.gazeteer_pth = NEL_PATHS[0][0]

    def predict(self, texts: list[str]) -> list[list[dict]]:
        return lookup_inference(texts, self.gazeteer_pth, self.case_sensitive)

class BiencoderPipeline(AnnotationPipeline):
    """Full pipeline: NER → NEL (dense retrieval)"""
    def __init__(self, agg_strat="first", device=None):
        self.agg_strat = agg_strat
        self.device = device

    def predict(self, texts: list[str]) -> list[list[dict]]:
        ner_results = ner_inference(texts, NER_PATHS, agg_strat=self.agg_strat, device=self.device)
        neg_results = ner_results.pop()
        norm_results = biencoder_inference(ner_results, NEL_PATHS, device=self.device)
        norm_results = join_all_entities(norm_results)
        final_results = add_negation_uncertainty_attributes(norm_results, neg_results)
        return final_results
    

# class FuzzyMatchPipeline(AnnotationPipeline):
#     def __init__(self, method = "jaro_winkler"):
#         self.method = method
    
#     def predict(self, texts: list[str]) -> list[list[dict]]:
#         # Apply fuzzy matching directly to raw text, no NER upstream
#         pass