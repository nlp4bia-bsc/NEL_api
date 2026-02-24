import re

SPLIT_WORDS = re.compile(
    r'([0-9A-Za-zÀ-ÖØ-öø-ÿ]+|[^0-9A-Za-zÀ-ÖØ-öø-ÿ])'
)
"""This regex splits text into tokens such that:
Words (including accented letters and numbers) stay grouped
Every non-alphanumeric character becomes its own token
"Hola, qué tal?" -> ["Hola", ",", " ", "qué", " ", "tal", "?"]
"""

def pretokenize_sentence(sentence: str) -> tuple[str, list[int]]:
    """Pretokenizes a sentence by splitting it into tokens and adding spaces between non-space tokens, and saves the added space positions for later alignment. 
    This is done to ensure that the token classification model can correctly identify entities that are attached to punctuation or other words without spaces.
    For example:
        "Hola, qué tal?" -> ["Hola", ",", " ", "qué", " ", "tal", "?"] ->  ["Hola ", ",", " ", "qué", " ", "tal ", "?"]  -> "Hola , qué tal ?"
    """
    pretokens = [t for t in SPLIT_WORDS.split(sentence) if t]
    added_spaces_pos = []
    char_ct = 0
    for i, (curr_token, next_token) in enumerate(zip(pretokens[:-1], pretokens[1:])):
        char_ct += len(curr_token)
        if not curr_token.isspace() and not next_token.isspace():
            # If both are non-space tokens, we insert a space between them
            pretokens[i] = curr_token + ' '
            added_spaces_pos.append(char_ct) # char count aligns with new space position
            char_ct += 1 # add that space to the character count

    return ''.join(pretokens), added_spaces_pos