from typing import Protocol
from abc import abstractmethod

from app.src.data_structures import Annotation, RecordMetadata, NlpResponse, NlpOutput, NlpServiceInfo

class DataFormatter(Protocol):
    @abstractmethod
    def serialize(self, text: str, annotations: list[dict], footer: dict) -> dict:
        pass

class Dt4h_nlp_cdm(DataFormatter):
    def serialize(self, text: str, annotations: list[dict], footer: dict) -> dict:
        validated_annotations = [Annotation(**self.rename_ann2cdm(a)) for a in annotations]

        record_metadata = RecordMetadata(
            **{k: footer.get(k) for k in RecordMetadata.model_fields if k in footer},
            text=text,
            nlp_processing_pipeline_name=self.__class__.__name__,
        )

        response = NlpResponse(
            nlp_output=NlpOutput(
                record_metadata=record_metadata,
                annotations=validated_annotations,
            ),
            nlp_service_info=NlpServiceInfo(
                service_model=self.__class__.__name__,
            )
        )

        return response.model_dump(mode="json")
    
    @staticmethod
    def rename_ann2cdm(ann: dict):
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