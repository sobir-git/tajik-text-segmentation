import datasets

from src.tokenizer import tokenize_text


class Annotated:
    
    _start_labels: "list[int]" = None
    _end_labels: "list[int]" = None
    _text: str = None

    def __init__(self, tokens, spans=None, start_labels=None, end_labels=None):

        if spans is None:
            self._start_labels = start_labels
            self._end_labels = end_labels
            assert len(tokens) == len(start_labels) == len(end_labels)
            spans = self.convert_binary_labels_to_spans(start_labels, end_labels)

        self.tokens: "list[str]" = tokens
        self.spans: "list[tuple[int,int]]" = spans

    def replace_token(self, token_idx, new_token):
        '''Replace a token with a new token. Return a new copy of the Annotated object.'''
        return Annotated(
            self.tokens[:token_idx] + [new_token] + self.tokens[token_idx+1:], spans=self.spans
        )

    def merge(self, other: "Annotated", separator=" "):
        '''Merge two annotated objects with a separator. The separater will be concatenated to the start of the 
        other's first token if it is not already present there.'''

        # insert the separator at the start of the other's first token
        other_first_token = separator + other.tokens[0].lstrip()

        # replace other's first token with the modified one
        other = other.replace_token(0, other_first_token)

        # offset other's spans by the number of tokens in the 'self.tokens'
        offset = len(self.tokens)
        other_spans = [(start + offset, end + offset) for start, end in other.spans]

        # create a new annotated object by merging the two and return it
        return Annotated(
            self.tokens + other.tokens, spans=self.spans + other_spans
        )
    
    def select_spans(self, start: int, end: int):
        '''Select a subset by a range of spans. Return a new Annotated object.'''
        new_spans = self.spans[start:end]
        assert new_spans, self
        
        # get the token range of the selected spans
        start_token_id = new_spans[0][0]  # first token_id of the first span
        assert len(new_spans[-1]) == 2, self
        end_token_id = new_spans[-1][1]   # last token_id of the last span
        new_tokens = self.tokens[start_token_id:end_token_id]

        # offset spans by subtracting the start of the first span
        new_spans = [(i-start_token_id,j-start_token_id) for i,j in new_spans]

        return Annotated(new_tokens, spans=new_spans)

    @property
    def text(self):
        if self._text is not None:
            return self._text

        self._text = ''.join(self.tokens)
        return self._text

    @property
    def start_labels(self):
        '''Return a binary array of start labels for each token'''
        if self._start_labels is not None:
            return self._start_labels
        
        labels = [0] * len(self.tokens)
        for start, _ in self.spans:
            labels[start] = 1
        
        self._start_labels = labels
        return labels

    @property
    def end_labels(self):
        '''Return a binary array of end labels for each token'''
        if self._end_labels is not None:
            return self._end_labels
        
        labels = [0] * len(self.tokens)
        for _, end in self.spans:
            labels[end - 1] = 1  # subtract one to make the label fall on the token index
        
        self._end_labels = labels
        return labels

    @staticmethod
    def convert_binary_labels_to_spans(start_labels, end_labels) -> "list[tuple[int,int]]":


        # Initialize a list to store the sentence spans
        spans = []
        
        # Initialize a list to store the current span
        cur_span = []
        
        # Iterate over the start labels and end labels simultaneously
        for i, (start, end) in enumerate(zip(start_labels, end_labels)):
            # Check if the current token is the start of a sentence
            if start:
                # If there is a current span, append it to the sentence spans list
                if cur_span:
                    spans.append(tuple(cur_span))
                
                # Start a new current span with the current index
                cur_span = [i]
            
            # Check if the current token is the end of a sentence
            if end:
                # Append the current index plus 1 to the current span, indicating the end of the sentence
                cur_span.append(i+1)
                
                # Append the current span to the sentence spans list
                spans.append(tuple(cur_span))

                # Initialize a new current span
                cur_span = []

        assert not cur_span

        # Quick check the validity of spans
        if not all(len(span) == 2 for span in spans):
            assert len(start_labels) == len(end_labels)
        
        return spans
    

    def __repr__(self):
        return f'''<Annotatated: {repr(self.annotated_text)}\nSpans: {self.spans}>'''

    @property
    def annotated_text(self):
        buffer = []
        assert len(self.start_labels) == len(self.end_labels) == len(self.tokens)
        for start_label, end_label, token in zip(self.start_labels, self.end_labels, self.tokens):
            prefix = '{{' if start_label else ''
            suffix = '}}' if end_label else ''
            buffer.append(f'{prefix}{token}{suffix}')
        return ''.join(buffer)

    def get_sentences(self):
        sentences = []
        for s,e in self.spans:
            sentences.append(''.join(self.tokens[s:e]))
        return sentences


def load_dataset():
    annotations = datasets.load_dataset('sobir-hf/tajik-text-segmentation', split='train')
    return list(annotations)


def tokenize_annotation(ann) -> Annotated:
    """
    Tokenize text and convert sentence spans from character-level to token-level.

    Args:
        ann (dict): Annotation dictionary containing 'text' and 'spans'.

    Returns:
        dict: Processed information including 'text', 'tokens', 'start_labels', 'end_labels',
              'token_spans', and 'spans'.
    """
    # Extract the 'text' and 'spans' from the input 'ann' dictionary
    text = ann['text']
    spans = ann['spans']
    
    # check validity of spans
    for span1, span2 in zip(spans[:-1], spans[1:]):
        assert 0 <= span1[0] < span1[1] <= span2[0] < span2[1] <= len(text), f"{span1} {span2}"

    # Sort the spans by their start indices to ensure they are in ascending order
    spans = sorted(spans, key=lambda x: x[0])
    
    # Tokenize the text and get the tokens and corresponding token spans
    tokens, token_spans = tokenize_text(text)

    # Check the validity of token_spans
    for span1, span2 in zip(token_spans[:-1], token_spans[1:]):
        assert 0 <= span1[0] < span1[1] == span2[0] < span2[1] <= len(text), f"{span1} {span2}"
    assert token_spans[-1][1] == len(text)

    # Get the start indices of each span
    start_indices = [span[0] for span in spans]

    # NOTE: assumes that span[1] always > 0
    # Get the end indices of each span
    end_indices = [span[1] - 1 for span in spans]

    # Get the labels for the start and end indices based on the token spans
    start_labels = get_labels(token_spans, start_indices)
    end_labels = get_labels(token_spans, end_indices)

    return Annotated(tokens, start_labels=start_labels, end_labels=end_labels)


def get_labels(spans, indices):
    """For each span checks if any indices belong to it.
    A span in a pair of integers a,b representing the interval [a, b).
    """

    labels = [0] * len(spans)

    i = 0  # token index
    for j in indices:
        # find token that contains start of span
        # while current token ends before start of span
        while i < len(spans) and spans[i][1] <= j:
            i += 1
        
        if i == len(spans):
            break  # reached the end of the text
        
        # check if current token contains the start of span
        if spans[i][0] <= j:
            labels[i] = 1

    return labels
