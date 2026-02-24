def align_results(results_pre: list[dict], added_spaces: list[int], start_sent_offset: int) -> list[dict]:
    """
    Realign NER results produced on pretokenized text back to the original text.
    - Fixes character offsets (start, end)
    - Removes artificially added spaces inside entity spans
    - Normalizes output field names
    """

    def spaces_before(pos: int) -> int:
        """Count artificial spaces that appear before a given position."""
        return sum(1 for space_pos in added_spaces if space_pos < pos)

    def spaces_inside_span(start: int, end: int) -> set[int]:
        """Return the set of artificial space positions strictly inside an entity span."""
        return {space_pos for space_pos in added_spaces if start < space_pos < end}

    def clean_surface_form(word: str, span_start: int, artificial_spaces: set[int]) -> str:
        """Strip leading/trailing whitespace and remove any artificial spaces from the word."""
        return "".join(
            char
            for i, char in enumerate(word.strip())
            if (i + span_start) not in artificial_spaces
        )

    aligned_results = []
    for ent in results_pre:
        start, end = ent["start"], ent["end"]

        artificial_spaces = spaces_inside_span(start, end)
        cleaned_word = clean_surface_form(ent["word"], start, artificial_spaces)

        aligned = ent.copy()
        aligned["span"] = cleaned_word
        aligned["ner_class"] = aligned.pop("entity_group")
        aligned["start"] = start_sent_offset + start - spaces_before(start)
        aligned["end"]   = start_sent_offset + end   - spaces_before(end)
        aligned.pop("word", None)

        aligned_results.append(aligned)

    return aligned_results

def join_all_entities(results: list[list[list[dict]]]) -> list[list[dict]]:
    num_texts = len(results[0])  # number of documents
    entities_all = []

    for text_idx in range(num_texts):
        entities_file = []
        for model_idx in range(len(results)):
            entities_file.extend(results[model_idx][text_idx])
        entities_file = sorted(entities_file, key=lambda x: (x['start'], -x['end']))
        entities_all.append(entities_file)
    return entities_all
