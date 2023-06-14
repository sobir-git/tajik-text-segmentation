import os
import torch
import torch.nn as nn
from src.dataset import load_dataset, tokenize_annotation
from src.nn_model.dataset import SentenceBoundaryDataset
from src.nn_model.model import SentenceBoundaryModel, batch_to_device

from src.nn_model.utils import Average, seed_everything
from src.nn_model.vocab import Vocabulary



class Checkpoint:
    def __init__(self, path, config: dict, model: nn.Module, vocab: Vocabulary):
        self.path = path
        self.config = config
        self.model = model
        self.vocab = vocab
        self.metrics = {}

        # ensure path directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
    
    def save(self):
        torch.save({
            'config': self.config,
            'model': self.model.state_dict(),
            'vocab': self.vocab.state_dict(),
            'metrics': self.metrics
        }, self.path)

    def load(self, device='cpu'):
        checkpoint = torch.load(self.path, map_location=device)
        self.config = checkpoint['config']
        self.metrics = checkpoint['metrics']
        self.model.load_state_dict(checkpoint['model'])
        self.vocab.load_state_dict(checkpoint['vocab'])

    @classmethod
    def init_from_path(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        # init model and vocab
        model = SentenceBoundaryModel(config)
        vocab = Vocabulary()
        
        # load state dicts
        model.load_state_dict(checkpoint['model'])
        vocab.load_state_dict(checkpoint['vocab'])

        return cls(path, config, model, vocab)


def train(model, optimizer, criterion, train_dataset, test_dataset, num_epochs=100, device='cpu', lr_scheduler=None, checkpoint: Checkpoint=None):
    # Set model to train mode
    model.train()
    
    # Calculate average sequence length
    sample_size = min(100, len(train_dataset))
    avg_seq_len = sum(len(train_dataset[i]['labels']) for i in range(sample_size)) / sample_size

    # keeping track of best performance for checkpointing
    best_score = 0
    
    # initialize a new average loss tracker
    avg_loss = Average()

    for epoch in range(num_epochs):
        for i, batch_idx in enumerate(torch.randperm(len(train_dataset))):
            
            sample = train_dataset[batch_idx]
            sample = batch_to_device(sample, device)

            labels = sample['labels']
            
            # Forward pass
            logits = model(sample)

            loss = criterion(logits, labels.float())
            
            # Backward and optimize
            optimizer.zero_grad()

            # scale loss by the sequence length
            loss_scaled = loss * labels.size(0) / avg_seq_len

            loss_scaled.backward()
            
            # clip gradient norms
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            optimizer.step()
            
            avg_loss.update(loss.item(), labels.size(0))

            # Logging time
            if i % min(90,  len(train_dataset)) == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch} -------------------------> TrainLoss: {avg_loss.get_value():.4f} lr: {lr:.5f}')
                
                # Log to wandb
                wandb.log({'train_loss': avg_loss.get_value(), 'lr': lr})

                # update scheduler
                if lr_scheduler:
                    lr_scheduler.step()
                
                # reset average loss
                avg_loss = Average()

                # evaluate and checkpoint
                val_loss, f1_score = eval(model, criterion, test_dataset, device)
                if f1_score > best_score:
                    if checkpoint:
                        checkpoint.save()
                        checkpoint.metrics['f1_score'] = f1_score
                    best_score = f1_score

                # set model back to train mode
                model.train()


@torch.no_grad()
def eval(model, criterion, dataset, device='cpu'):
    model.eval()  # set the model to evaluation mode
    error_rates = Average()
    avg_loss = Average()

    precisions = Average()
    recalls = Average()
    f1_scores = Average()

    for batch_idx in range(len(dataset)):
        
        sample = dataset[batch_idx]
        sample = batch_to_device(sample, device)

        labels = sample['labels']
        
        # Forward pass
        logits = model(sample)

        loss = criterion(logits, labels.float())
        
        avg_loss.update(loss, labels.size(0))

        predicted = logits > 0
        error_rates.update((predicted != labels).sum(0) / labels.size(0), labels.size(0))

        tp = torch.sum((predicted) & (predicted == labels), 0)
        tn = torch.sum((~predicted) & (predicted == labels), 0)
        fp = torch.sum((predicted) & (predicted != labels), 0)
        fn = torch.sum((~predicted) & (predicted != labels), 0)
        precisions.update(tp / ((tp + fp) + ((tp + fp) == 0) ), (tp + fp))
        recalls.update(tp / (fn + tp + ((fn + tp) == 0)), (fn + tp))
        f1_scores.update(2*tp / (2*tp + fp + fn), (2*tp + fp + fn))

    print(f"\n[Validation] Loss: {avg_loss.get_value():.4f}, "
          f"F1[Start]: {f1_scores.get_value()[0]:.3f}, "
          f"F1[End]: {f1_scores.get_value()[1]:.3f}, "
          f"ErrorRate[Start]: {error_rates.get_value()[0]:.3f}, "
          f"ErrorRate[End]: {error_rates.get_value()[1]:.3f}, "
          f"Prec[Start]: {precisions.get_value()[0]:.3f}, "
          f"Prec[End]: {precisions.get_value()[1]:.3f}, "
          f"Rec[Start]: {recalls.get_value()[0]:.3f}, "
          f"Rec[End]: {recalls.get_value()[1]:.3f}\n"  
          )
    
    wandb.log({
        'Validation': {
            'Loss': avg_loss.get_value(),
            'F1': f1_scores.get_value().mean().item(),
            'F1[Start]': f1_scores.get_value()[0],
            'F1[End]': f1_scores.get_value()[1],
            'ErrorRate[Start]': error_rates.get_value()[0],
            'ErrorRate[End]': error_rates.get_value()[1],
            'Prec[Start]': precisions.get_value()[0],
            'Prec[End]': precisions.get_value()[1],
            'Rec[Start]': recalls.get_value()[0],
            'Rec[End]': recalls.get_value()[1]
        }
        
    })
    
    return avg_loss.get_value(), f1_scores.get_value().mean().item()


def run_training():
    seed_everything(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Loading dataset...')
    annotations = load_dataset()
    
    print('Tokenizing dataset...')
    annotations = [tokenize_annotation(ann) for ann in annotations]

    print('Creating vocabulary...')
    vocab = Vocabulary()

    config = {
        'input_size': len(vocab),
        'emb_size': 48,
        'num_layers': 1,
        'hidden_size': 32,
        'n_tags': len(vocab.tagnames),
        'output_size': 2,
        'dropout': 0.5,
    }
    wandb.init(config=config)

    # Create model, optimizer, criterion
    criterion = nn.BCEWithLogitsLoss()
    model = SentenceBoundaryModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.000001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 60, 1)
    
    # split dataset into train and test
    train_size = int(len(annotations) * 0.8)
    test_size = len(annotations) - train_size
    indices = torch.randperm(train_size+test_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_annotations = [annotations[i] for i in indices[:train_size]]
    test_annotations = [annotations[i] for i in indices[train_size:]]

    # Create shuffled/mixed dataset
    WINDOW_SIZE=3
    N_SENTENCES=8
    # train_dataset = SentenceBoundaryDataset(train_annotations, vocab, shuffle=False)
    train_dataset = SentenceBoundaryDataset(train_annotations, vocab, shuffle=True, window_size=WINDOW_SIZE, batch_size=N_SENTENCES)
    test_dataset = SentenceBoundaryDataset(test_annotations, vocab, shuffle=False)
    print('train_size:', train_size)
    print('test_size:', test_size)

    checkpoint = Checkpoint('checkpoints/checkpoint.pt', config, model, vocab)

    # Launch training
    train(model, optimizer, criterion, train_dataset, test_dataset, device=device, lr_scheduler=lr_scheduler, checkpoint=checkpoint)


if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()
    import wandb

    run_training()
