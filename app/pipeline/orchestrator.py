from app.pipeline.ner import ner_inference
from app.pipeline.nel import nel_inference
from app.pipeline.negation import add_negation_uncertainty_attributes
from app.config import NER_PATHS, NEL_PATHS
from app.pipeline.abs_model import ModelAnnotation
from app.utils.results_postprocessing import join_all_entities


class NERL_Orchestrator(ModelAnnotation):
    def __init__(self, agg_strat: str = "first", device: str | None = None):
        self.agg_strat = agg_strat
        self.device = device

    def predict(self, texts: list[str], footers: list[dict]) -> list[dict]:
        ner_results = ner_inference(texts, NER_PATHS, agg_strat=self.agg_strat, device=self.device)
        neg_results = ner_results.pop()
        norm_results = nel_inference(ner_results, NEL_PATHS, device=self.device)
        norm_results = join_all_entities(norm_results)
        final_results = add_negation_uncertainty_attributes(norm_results, neg_results)
        std_results = [[self.rename_ann2cdm(ann) for ann in annotated_doc] for annotated_doc in final_results]
        return [self.serialize(text, ann_result, footer) for text, ann_result, footer in zip(texts, std_results, footers)]

    def rename_ann2cdm(self, ann: dict):
        """Converts annoatations' fields into the CDM V2 variables.
        The current output is the following:
            {
                "start": int,
                "end": int,
                "ner_score": float,
                "span": str,
                "ner_class": str,
                "code": str,
                "term": str,
                "nel_score": float,
                "is_negated": bool,
                "negation_score": float,
                "is_uncertain": bool,
                "uncertainty_score": float
            }

        And the intended one has the following structure:
            {
                "concept_class": "symptom | disorder/disease | procedure | medication | cardiology entity | other", (ner_class, convert "ENFERMEDAD": "disorder/disease", "SINTOMA": "symptom")
                "start_offset": int, (start)
                "end_offset": int, (end)
                "concept_mention_string": "str", (span)
                "extraction_confidence": float, (ner_score)
                "concept_str": str, (term)
                "concept_code": str, (code)
                "concept_confidence": float, (nel_score)
                "negation": "str", (convert is_negated to yes | no)
                "negation_confidence": float, (negation_score)
                "uncertainty": "str", (convert is_uncertain to yes | no)
                "uncertainty_confidence": float (uncertainty_score)
            }
        """
        ner_class_map = {
            "ENFERMEDAD": "disorder/disease",
            "SINTOMA": "symptom",
        }

        try:
            concept_class = ner_class_map.get(ann["ner_class"], ann["ner_class"].lower())

            return {
                "concept_class": concept_class,
                "start_offset": ann["start"],
                "end_offset": ann["end"],
                "mention_string": ann["span"],
                "extraction_confidence": ann["ner_score"],
                "concept_str": ann["term"],
                "concept_code": ann["code"],
                "concept_confidence": ann["nel_score"],
                "negation": "yes" if ann["is_negated"] else "no",
                "negation_confidence": ann["negation_score"],
                "uncertainty": "yes" if ann["is_uncertain"] else "no",
                "uncertainty_confidence": ann["uncertainty_score"],
            }

        except KeyError as e:
            raise ValueError(f"Missing expected annotation field: {e}")