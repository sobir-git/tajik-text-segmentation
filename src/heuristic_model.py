import re

import numpy as np


class Features:
    def __init__(self, token) -> None:
        # remove quotations and certain characters
        token = re.sub(r'[»«”“"’‘]','', token, re.UNICODE)

        # normalize token
        token = token.replace('млн.', 'миллион')
        token = token.replace('млрд.', 'миллиард')
        token = token.replace('ҳаз.', 'ҳазор')
        token = token.replace('ғ.', 'ғайра')
        token = token.replace(' \n', '\n')
        token = token.replace('\n ', '\n')
        
        token_strip = token.strip()
        self.is_title = token_strip.istitle()
        self.is_upper = token.isupper()
        self.is_lower = token.islower()
        self.is_digit = token.isdigit()
        self.is_empty = len(token) == 0
        self.starts_with_space = token.startswith(' ')
        self.starts_with_linebreak = token.startswith('\n')
        self.is_parantheses_open = re.match(r'\s?[\[\(\{]', token, re.DOTALL) is not None
        self.is_parantheses_close = re.match(r'.*[\]\)\}]', token, re.DOTALL) is not None
        self.endswith_dot_and_others = re.match(r'.*[.?!;:…]+[\W]?$', token, re.DOTALL) is not None
        self.endswith_colon = token.endswith(':')
        self.endswith_semicolor = token.endswith(';')
        self.ends_with_three_dots = token.endswith('…') or token.endswith('...')
        self.is_acronym = token.isupper()
        self.non_letter = re.match(r'^[\W|\d]+$', token, re.DOTALL) is not None
        self.token = token
        self.is_day = re.match(r'\s?\d{1,2}-.\w+$', token, re.DOTALL) is not None
        self.is_punct = re.match(r'\s?[.?!;:]+$', token, re.DOTALL) is not None
        self.is_dash = re.match(r'\s?[-–—]', token, re.DOTALL) is not None

    def __bool__(self):
        return not self.is_empty
    
    def __repr__(self) -> str:
        return f'Features({repr(self.token)})'


def predict_token(prefix, cur, suffix):
    """Give a token and its prefix and suffix tokens, determine if the token is 1) start of sentence, 2) end of sentence"""
    start, end = False, False

    cur = Features(cur)
    prefix = Features(prefix)
    suffix = Features(suffix)


    if cur.is_title or cur.starts_with_linebreak or prefix.endswith_dot_and_others:
        start = True

    if cur.endswith_dot_and_others:
        end = True
    
    if suffix.starts_with_linebreak:
        end = True

    if prefix.endswith_dot_and_others:
        start = True
    
    if prefix.is_acronym:
        start = False

    if cur.is_lower and not cur.is_day:
        start = False

    if suffix and suffix.is_lower and not suffix.non_letter:
        end = False

    if cur.non_letter and (cur.endswith_dot_and_others or cur.is_parantheses_close) and not cur.is_punct:  #  " 1."
        end = False

    if cur.is_dash and cur.is_lower:
        start = False

    if suffix.is_dash and suffix.is_lower:
        end = False

    if prefix and prefix.is_acronym:
        start = False
    
    if cur.is_acronym:
        end = False
    
    if prefix.starts_with_linebreak and prefix.non_letter:
        start = False

    if prefix and not prefix.endswith_dot_and_others:
        start = False

    if cur.is_parantheses_close and not cur.is_parantheses_open:
        start = False

    if cur.is_parantheses_open and not cur.is_parantheses_close:
        end = False

    if prefix.is_parantheses_open and not prefix.is_parantheses_close:
        start = False

    if suffix.is_parantheses_close and not suffix.is_parantheses_open:
        end = False

    if cur.starts_with_linebreak:
        start = True

    if suffix.starts_with_linebreak:
        end = True

    if not prefix:
        start = True
    if not suffix:
        end = True
    
    return start, end


class HeuristicModel:
    def predict(self, tokens):
        predictions = []
        # iterate over all 3-tuple of consecutive tokens
        for prefix, cur, suffix in zip([''] + tokens, tokens, tokens[1:] + ['']):
            start, end = predict_token(prefix, cur, suffix)
            predictions.append((start, end))
        
        predictions = np.array(predictions, dtype=np.int32)
        return predictions
