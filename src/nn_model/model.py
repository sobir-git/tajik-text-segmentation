import copy
import torch 
import torch.nn as nn

from src.dataset import load_dataset, tokenize_annotation
from src.nn_model.dataset import SentenceBoundaryDataset
from src.nn_model.vocab import Vocabulary


class BOWEmbedding(nn.Module):
    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__()
        # make a copy of config and update it by kwargs
        config = copy.deepcopy(config)
        config.update(**kwargs)
        emb_size = config['emb_size']
        n_tags = config['n_tags']

        self.embedding = nn.Embedding(config['input_size'], emb_size)
        self.pos_emb = nn.Linear(2, emb_size, bias=False)
        self.dropout = nn.Dropout(config['dropout'])
        self.conv1 = nn.Conv1d(emb_size, emb_size, kernel_size=3, padding=1)
        self.tag_weights = nn.Parameter(torch.ones((1, n_tags))  / n_tags)
        self.tag_emb = nn.Parameter(torch.randn(n_tags, emb_size))
        self.config = config
    
    def forward(self, sample):
        sizes = sample['sizes']  # (n_tokens,)  -> token sizes
        char_enc = sample['char_enc']  # (n_tokens, max_token_size)
        tag_enc = sample['tag_enc'].float()  # (n_tokens, max_token_size, n_tags)

        char_emb = self.embedding(char_enc)  # (n_tokens, max_token_size, emb_size)
        
        # zero out paddings
        mask = torch.arange(0, char_enc.size(1), device=sizes.device).unsqueeze(0) < sizes.unsqueeze(1)  # (n_tokens, max_token_size)
        char_emb *= mask.unsqueeze(2).float()
        
        tag_enc = self.tag_weights * tag_enc

        tag_emb = torch.matmul(tag_enc, self.tag_emb.unsqueeze(0))

        out = char_emb + tag_emb

        out = self.dropout(out)

        out = out.transpose(1,2)
        out = self.conv1(out)
        out = torch.relu(out)
        
        out = out.transpose(1,2)
        out = out.max(1)[0]
        out = self.dropout(out) if self.dropout else out
        return out


class SentenceBoundaryModel(nn.Module):
    def __init__(self, config: dict, **kwargs) -> None:
        super(SentenceBoundaryModel, self).__init__()
        # make a copy of config and update it by kwargs
        config = copy.deepcopy(config)
        config.update(**kwargs)
        self.embedding = BOWEmbedding(config)
        self.gru = nn.GRU(config['emb_size'], config['hidden_size'], config['num_layers'], batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config['hidden_size'] * 2, config['output_size'])  # times 2 because of bidirectionality
        self.config = config

    def forward(self, sample):
        x = self.embedding(sample)
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


class NNPredictor:
    def __init__(self, model: nn.Module, vocab: Vocabulary) -> None:
        self.model = model
        self.vocab = vocab

    def predict(self, tokens: "list[str]"):
        x = self.vocab.encode_tokens(tokens)
        logits = self.model(x)
        preds = logits > 0
        return preds


def batch_to_device(batch, device):
    for k in ['char_enc', 'tag_enc', 'labels', 'sizes']:
        batch[k] = batch[k].to(device)
    return batch


if __name__ == '__main__':
    device = 'cpu'

    print('Loading dataset...')
    annotations = load_dataset()
    
    print('Tokenizing dataset...')
    annotations = [tokenize_annotation(ann) for ann in annotations]

    print('Creating vocabulary...')
    vocab = Vocabulary()

    dataset = SentenceBoundaryDataset(annotations, vocab, shuffle=True)
    sample = dataset[0]
    print('Sample:', sample)

    config = {
        'input_size': len(vocab),
        'emb_size': 48,
        'hidden_size': 32,
        'num_layers': 1,
        'n_tags': len(vocab.tagnames),
        'output_size': 2,
        'dropout': 0.5,
    }

    model = SentenceBoundaryModel(config)
    
    sample = batch_to_device(sample, device)
    out = model(sample)
    print('out.shape:', out.shape)
