import re


def tokenize_text(text):
    # split text into tokens using regular expression pattern
    regex = r"[\s\-–—«“‘\[\(]*[\S]+(\s*[»”;?!’\)\]:,.\%]+)?"

    matches = re.finditer(regex, text, re.MULTILINE | re.UNICODE)

    tokens: "list[str]" = []
    spans: "list[tuple[int,int]]" = []
        
    for match in matches:
        start = match.start()
        end = match.end()
        match = match.group()
        tokens.append(text[start:end])
        spans.append((start, end))

    return tokens, spans
