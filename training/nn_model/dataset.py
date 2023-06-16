import random
from typing import List
from torch.utils.data import Dataset
import torch
from tajik_text_segmentation.annotated import Annotated, load_dataset, tokenize_annotation
from tajik_text_segmentation.heuristic_model import HeuristicModel

from tajik_text_segmentation.nn_model.vocab import Vocabulary


class SentenceBoundaryDataset(Dataset):
    def __init__(self, annotations: List[Annotated], vocab: Vocabulary, shuffle:bool=False, loops: int=100, batch_size:int=3, window_size:int=3, sep:str='   \n', heuristic_model: HeuristicModel=None):

        if shuffle:
            # arrange all sentences in a sequence
            # a sentence is identified by the annotated text and its span index in that text
            sentences = []
            for ann in annotations:
                for i in range(len(ann.spans)):
                    sentences.append((i, ann))
            
            # repeat to account for batches and epochs (loops)
            sentences = sentences * batch_size * loops

            random.shuffle(sentences)

            self.sentences = sentences
            self.batch_size = batch_size
            self.window_size = window_size
        
        self.vocab = vocab
        self.sep = sep
        self.shuffle = shuffle
        self.annotated_texts = annotations
        self.heuristic_model = heuristic_model

    def __len__(self):
        '''Return the 'number of samples' of the dataset.'''
        if not self.shuffle:
            return len(self.annotated_texts)
        
        return len(self.sentences) // self.batch_size

    def verify_item(self, item: Annotated):
        '''Verify that the item is correct.'''
        for start, end in item.spans:
            assert end > start, item
            assert end - 1 < len(item.tokens), item

    def select_neighbours(self, item: Annotated, sentence_id: int, n_neighbors: int):
        '''Select neighbours of a sentence in the original text.'''
        n_neighbors = random.randint(1,n_neighbors)
        n_sentences = len(item.spans)

        # select first and last neighbour sentences
        first_id = max(0, sentence_id - n_neighbors//2)
        last_id = min(first_id + n_neighbors, n_sentences)
        if last_id - first_id < n_neighbors:
            # try moving start to left
            first_id = max(0, last_id - n_neighbors)
        
        assert last_id > first_id, item

        # return the selected neighbours as a new item
        return item.select_spans(first_id, last_id)
    
    def modify_last_span(self, item: Annotated):
        """Make last span end with period or other ending characters."""

        token_idx = len(item.tokens) - 1

        # get last token
        last_token = item.tokens[token_idx]

        ok_ending_chars = '.!?'

        if last_token[-1] in ok_ending_chars:
            # return item without modification
            return item

        # remove any of these characters from the end: ,:;
        i = len(last_token)
        while i > 0 and last_token[i-1] in ',:;':
            i -= 1
        
        last_token = last_token[:i]

        if not last_token:
            # it was truncated completely... this must be rare
            return item  # no modification

        if last_token[-1] not in ok_ending_chars:
            # add a period
            last_token = last_token + '.'

        return item.replace_token(token_idx, last_token)

    def __getitem__(self, idx):
        if not self.shuffle:
            item = self.annotated_texts[idx]
            return self.prepare_item(item)
        
        # select sentences from the pool
        i = idx*self.batch_size

        n_sentences = random.randint(1, self.batch_size)
        sentences = self.sentences[i:i+n_sentences]
        
        # get surrounding sentences in their context
        items = [self.select_neighbours(sent[1], sent[0], self.window_size) for sent in sentences]
        items = [self.modify_last_span(item) for item in items]

        # join all items
        item = items[0]
        for item_other in items[1:]:
            # select a separator at random
            sep = random.choice(self.sep)
            item = item.merge(item_other, separator=sep)

        return self.prepare_item(item)

    def prepare_item(self, item: Annotated):
        '''Prepares item for training.'''

        pack = self.vocab.encode_tokens(item.tokens)
        if self.heuristic_model:
            heuristic_predictions = self.heuristic_model.predict(item.tokens)
            pack['heuristic_predictions'] = torch.tensor(heuristic_predictions, dtype=torch.float)
        pack['labels'] = torch.tensor([item.start_labels, item.end_labels], dtype=torch.long).T  # (n_tokens, 2)
        pack['item'] = item

        return pack


if __name__ == '__main__':
    print('Loading dataset...')
    annotations = load_dataset()

    print('Creating vocabulary...')
    vocab = Vocabulary()
    vocab.build_from_texts([ann['text'] for ann in annotations])
    
    # tokenize dataset
    print('Tokenizing dataset...')
    annotations = [tokenize_annotation(ann) for ann in annotations]
    # print(annotations[5])
    # print(annotations[5].select_spans(1,5))
    # print(annotations[5].select_spans(1,3).merge(annotations[5].select_spans(3,5), separator=' '))

    dataset = SentenceBoundaryDataset(annotations, vocab, shuffle=True)
    print(dataset[6])
