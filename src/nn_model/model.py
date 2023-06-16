import copy
import torch 
import torch.nn as nn

from src.dataset import load_dataset, tokenize_annotation
from src.heuristic_model import Features, HeuristicModel
from src.nn_model.boundary_resolver import SentenceBoundaryResolver
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
        self.fc = nn.Linear(config['hidden_size'] * 2 + config['n_features'] + (2 if config['use_heuristic_predictions'] else 0), config['output_size'])  # times 2 because of bidirectionality
        self.ft = nn.Identity()
        if config['feature_transform']:
            self.ft = nn.Sequential(*[
                nn.Linear(config['n_features'], config['n_features']),
                nn.PReLU()
            ])
        self.config = config

    def forward(self, sample):
        x = self.embedding(sample)
        out, _ = self.gru(x)
        if self.config['n_features']:
            features = sample['features'].float()
            features = self.ft(features)
            out = torch.cat([out, features], -1)
        if self.config['use_heuristic_predictions']:
            out = torch.cat([out, sample['heuristic_predictions']], -1)

        out = self.fc(out)
        return out


class NNPredictor:
    def __init__(self, config, model: nn.Module, vocab: Vocabulary, resolver: SentenceBoundaryResolver) -> None:
        self.model = model
        self.vocab = vocab
        self.resolver = resolver
        self.heuristic_model = HeuristicModel() if config['use_heuristic_predictions'] else None

    @torch.no_grad()
    def predict(self, tokens: "list[str]"):
        self.model.eval()
        pack = self.vocab.encode_tokens(tokens)
        if self.heuristic_model:
            heuristic_predictions = self.heuristic_model.predict(tokens)
            pack['heuristic_predictions'] = torch.tensor(heuristic_predictions, dtype=torch.float)
        logits = self.model(pack)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = self.resolver.resolve(probs, binarize_output=True)
        return preds


def batch_to_device(batch: dict, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()
    }


if __name__ == '__main__':
    device = 'cpu'

    print('Loading dataset...')
    annotations = load_dataset()
    
    print('Tokenizing dataset...')
    annotations = [tokenize_annotation(ann) for ann in annotations]

    print('Creating vocabulary...')
    vocab = Vocabulary()



    config = {
        'seed': 0,
        'input_size': len(vocab),
        'emb_size': 48,
        'num_layers': 1,
        'hidden_size': 32,
        'n_tags': len(vocab.tagnames),
        'output_size': 2,
        'dropout': 0.5,
        'optimizer_kwargs': {'lr': 0.003, 'weight_decay': 0.00001},
        'lr_scheduler': 'CosineAnnealingWarmRestarts',
        'lr_scheduler_kwargs': {'T_0': 60},
        # 'lr_scheduler_kwargs': {'T_0': 20},
        # 'lr_scheduler': 'StepLR',
        # 'lr_scheduler_kwargs': {'step_size': 50,'gamma': 0.1},
        # 'dataset_kwargs': {'shuffle': False}, 
        'dataset_kwargs': {'shuffle': True, 'window_size': 3, 'batch_size': 8}, 
        'n_features': Features.get_n_features(),
        'feature_transform': True,
        'use_heuristic_predictions': True
    }

    heuristic_model = HeuristicModel() if config['use_heuristic_predictions'] else None
    dataset = SentenceBoundaryDataset(annotations, vocab, shuffle=True, heuristic_model=heuristic_model)
    sample = dataset[0]
    print('Sample:', sample)

    model = SentenceBoundaryModel(config)
    
    sample = batch_to_device(sample, device)
    out = model(sample)
    print('out.shape:', out.shape)
