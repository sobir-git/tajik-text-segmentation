import re


def tokenize_text(text):
    # split text into tokens using regular expression pattern
    regex = r"[\s\-–—«“‘\[\(]*[\S]+(\s*[»”;?!’\)\]:,.\%]+)?"

    matches = re.finditer(regex, text, re.MULTILINE | re.UNICODE)

    tokens = []
    spans = []
        
    for match in matches:
        start = match.start()
        end = match.end()
        match = match.group()
        tokens.append(text[start:end])
        spans.append((start, end))

    return tokens, spans
