import unicodedata
import pandas as pd
from flashtext import KeywordProcessor


class LookUpMethod:
    def __init__(self, gaz_pth: str, case_sensitive: bool):
        self.case_sensitive = case_sensitive
        
        # load gazeteer
        self.gazeteer = pd.read_csv(gaz_pth, sep='\t').drop_duplicates(subset=["term"])
        
        # normalize ontology
        clean_terms = self.gazeteer['term'].astype(str).apply(self._normalize).to_list()

        # create lookup dict
        self.term_to_info = {
            clean: (original, code)
            for clean, original, code in zip(
                clean_terms, 
                self.gazeteer['term'], 
                self.gazeteer['code'])
        }
        
        # build and populate processor engine
        self.keyword_processor = KeywordProcessor(case_sensitive = case_sensitive)
        self.keyword_processor.add_keywords_from_list(clean_terms)
                    
    def _normalize(self, text: str) -> str:
        if not self.case_sensitive:
            text = text.lower()
            
        text = unicodedata.normalize('NFD', text)
        return "".join(c for c in text if unicodedata.category(c) != 'Mn')
        
    def run_lookup(self, text: str) -> list[dict]:
        # normalize text
        norm_text = self._normalize(text)
        
        matches = self.keyword_processor.extract_keywords(norm_text, span_info=True)
        results = []
        
        for matched_term, start, end in matches:
            original_term, code = self.term_to_info.get(matched_term, (matched_term, "NO_MAP"))
            results.append({
                "start": start,
                "end": end,
                "span": text[start:end],
                "ner_class": "LOOKUP",
                "ner_score": 1.0,
                "code": code,
                "term": original_term,
                "nel_score": 1.0,
            })
        return results


def lookup_inference(texts: list[str], gaz_pth: str, case_sensitive: bool) -> list[list[dict]]:  
    lookup_engine = LookUpMethod(gaz_pth, case_sensitive)

    results = []
    for text in texts:
        text_results = []
        text_results.extend(lookup_engine.run_lookup(text))
        #text_results.sort(key=lambda x: (x['start'], -x['end']))
        results.append(text_results)
        
    return results