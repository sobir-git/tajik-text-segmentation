from collections import Counter
import string

import torch

from src.dataset import load_dataset


class Vocabulary:
    '''Create character vocabulary from text corpus'''

    def __init__(self) -> None:

        self.punctuations = ''.join(['?', '«', '»', ':', '.', ',', '—', '!', '-', ';', '–', '…', '”', '“', '"', '@', '№', '=', '%', '’', '‘']) + string.punctuation
        self.tagnames = ('PUNCT', 'UPPER', 'PERIOD', 'CYRILLIC', 'LATIN', 'ARABIC', 'OPENING', 'CLOSING')
        self.tag_to_id = {t: i for i, t in enumerate(self.tagnames)}
        self.symbols = ['UNK', 'LETTER_LOWER', 'LETTER_UPPER', 'WHITESPACE', 'LINEBREAK', 'DIGIT']
        self.symbol_to_id = {c: i for i, c in enumerate(self.symbols)}

    def state_dict(self):
        return {
            'symbols': self.symbols,
            'symbol_to_id': self.symbol_to_id,
            'tag_to_id': self.tag_to_id,
            'tagnames': self.tagnames,
            'punctuations': self.punctuations
        }

    def load_state_dict(self, state_dict):
        self.symbols = state_dict['symbols']
        self.symbol_to_id = state_dict['symbol_to_id']
        self.tag_to_id = state_dict['tag_to_id']
        self.tagnames = state_dict['tagnames']
        self.punctuations = state_dict['punctuations']

    def tag_characters(self, chars: str) -> "list[list[bool]]":
        tags = []
        tag_to_id = self.tag_to_id

        for char in chars:
            tag_list = [False] * len(self.tagnames)
            # Add logic to determine the tag for character t
            # For example, if t is an uppercase letter, set 'UPPER' tag to 1
            if char.isupper():
                tag_list[tag_to_id['UPPER']] = True
            # If t is a period, set 'PERIOD' tag to 1
            if char == '.':
                tag_list[tag_to_id['PERIOD']] = True
            # If t is a Cyrillic character, set 'CYRILLIC' tag to 1
            if char.isalpha() and 1279 > ord(char) > 127:
                tag_list[tag_to_id['CYRILLIC']] = True
            # If t is a Latin character, set 'LATIN' tag to 1
            if char.isalpha() and ord(char) < 128:
                tag_list[tag_to_id['LATIN']] = True
            # If t is an Arabic character, set 'ARABIC' tag to 1
            if char.isalpha() and 1536 <= ord(char):
                tag_list[tag_to_id['ARABIC']] = True
            if char in '([{':
                tag_list[tag_to_id['OPENING']] = True
            if char in ')]}':
                tag_list[tag_to_id['CLOSING']] = True
            # If t is a punctuation mark, set 'PUNCT' tag to 1
            if char in self.punctuations:
                tag_list[tag_to_id['PUNCT']] = True
            tags.append(tag_list)
        
        return tags

    def build_from_texts(self, texts, threshold=2):
        charcounter = Counter()
        for text in texts:
            charcounter.update(set(text)) 

        # add all non-letter characters that appear at least in 
        # sufficient number of files to the vocabulary
        for char, count in charcounter.items():
            if count >= threshold and not char.isalnum() and not char.isspace() and not char.isdigit():
                self.symbols.append(char)
        
        # update id mappings
        self.symbol_to_id = {c: i for i, c in enumerate(self.symbols)}

    def encode_text(self, text: str, return_tensors=False):
        ids = []
        for char in text:
            if char.isalpha() and char.islower():
                ids.append(self.symbol_to_id['LETTER_LOWER'])
            elif char.isalpha() and char.isupper():
                ids.append(self.symbol_to_id['LETTER_UPPER'])
            elif char.isdigit():
                ids.append(self.symbol_to_id['DIGIT'])
            elif char == ' ':
                ids.append(self.symbol_to_id['WHITESPACE'])
            elif char == '\n':
                ids.append(self.symbol_to_id['LINEBREAK'])
            else:
                ids.append(self.symbol_to_id.get(char, self.symbol_to_id['UNK']))
        
        tags = self.tag_characters(text)

        if return_tensors:
            # convert to tensor
            ids = torch.LongTensor(ids)
            tags = torch.LongTensor(tags)

        return ids, tags

    def encode_tokens(self, tokens: "list[str]"):
        '''Encodes tokens to be readily input to model.'''
        sizes = torch.tensor([len(token) for token in tokens], dtype=torch.long)  # (n_tokens,)
        char_enc, tag_enc = zip(*(self.encode_text(token, return_tensors=True) for token in tokens))
        char_enc = torch.nn.utils.rnn.pad_sequence(char_enc, batch_first=True)  # (n_tokens, max_token_size)

        # tag_encodings is a list of arrays of size (token_size, n_tags)
        tag_enc = torch.nn.utils.rnn.pad_sequence(tag_enc, batch_first=True)  # (n_tokens, max_token_size, n_tags)

        return {
            'char_enc': char_enc,
            'tag_enc': tag_enc,
            'sizes': sizes
        }



    def __len__(self):
        return len(self.symbols)


if __name__ == '__main__':
    vocab = Vocabulary()

    text = '«Ss.Ҳӣ,پرچم123()stánΕθνικ'
    for char, tags in zip(text, vocab.tag_characters(text)):
        print(char, ', '.join([vocab.tagnames[i] for i,j in enumerate(tags) if j]))

    # =====================
    annotations = load_dataset()

    texts = [ann['text'] for ann in annotations]
    vocab.build_from_texts(texts)

    print("Vocabulary symbol set", vocab.symbols)
    print(vocab.encode_text('''
    Елизаветаи II (англисӣ: Elizabeth II; 21 апрел 1926[1][2][3][…], Mayfair[d], Ноҳияи Вестминстер[4] — 8 сентябр 2022[5][6], Balmoral Castle[d], Абердиншир[d][5][6]) — маликаи подшоҳии Бритониёи Кабир. 
    Елизавета дар Лондон Мэйфер, 21 апрели соли 1926 дар оилаи шоҳ Георги VI ва малика Елизаветаи Боуз-Лайон ба дунё омадааст ва 8 сентябри соли 2022 дар Искотланд вафот кард.
    '''))
