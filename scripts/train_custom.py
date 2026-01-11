import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from custom_transformer import TransformerModel
from dataset_loader import get_akkadian_dataset
import sentencepiece as spm
import argparse
import os
import time
import math
from tqdm import tqdm
import json
import shutil

# Optional: Adafactor
try:
    from transformers.optimization import Adafactor
    ADAFACTOR_AVAILABLE = True
except ImportError:
    ADAFACTOR_AVAILABLE = False


def train(model, loader, optimizer, criterion, device, pad_token_id, scheduler=None, gradient_accumulation_steps=1):
    """Train for one epoch with gradient accumulation support."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    accumulated_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        src = batch['src_ids'].to(device)
        tgt = batch['tgt_ids'].to(device)
        
        # Target input (for decoder) is shifted right (exclude last token)
        tgt_input = tgt[:, :-1]
        
        # Target output (for loss) is shifted left (exclude first token)
        tgt_output = tgt[:, 1:]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(src, tgt_input, pad_token_id)
        
        logits = model(
            src, 
            tgt_input, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask, 
            src_padding_mask=src_padding_mask, 
            tgt_padding_mask=tgt_padding_mask
        )
        
        # Reshape for loss
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        accumulated_loss += loss.item()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss * gradient_accumulation_steps
            accumulated_loss = 0
    
    # Handle remaining accumulated gradients
    if accumulated_loss > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss * gradient_accumulation_steps
        
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, pad_token_id):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            src = batch['src_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(src, tgt_input, pad_token_id)
            
            logits = model(
                src, 
                tgt_input, 
                src_mask=src_mask, 
                tgt_mask=tgt_mask, 
                src_padding_mask=src_padding_mask, 
                tgt_padding_mask=tgt_padding_mask
            )
            
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()
            
    return total_loss / len(loader)


class SPTokenizerCollator:
    """Collator using SentencePiece tokenizer."""
    def __init__(self, sp_model_path, max_len=128):
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.max_len = max_len
        self.pad_id = self.sp.pad_id()
        self.eos_id = self.sp.eos_id()
        self.bos_id = self.sp.bos_id()

    def __call__(self, batch):
        inputs = [item['transliteration_cleaned'] for item in batch]
        targets = [item['translation_normalized'] for item in batch]
        
        input_ids = [self.sp.encode(text, out_type=int) for text in inputs]
        target_ids = [self.sp.encode(text, out_type=int) for text in targets]
        
        def process_ids(ids_list, add_bos=False, add_eos=True):
            processed = []
            for ids in ids_list:
                if len(ids) > self.max_len - (1 if add_bos else 0) - (1 if add_eos else 0):
                    ids = ids[:self.max_len - (1 if add_bos else 0) - (1 if add_eos else 0)]
                
                if add_bos:
                    ids = [self.bos_id] + ids
                if add_eos:
                    ids = ids + [self.eos_id]
                    
                if len(ids) < self.max_len:
                    ids = ids + [self.pad_id] * (self.max_len - len(ids))
                
                processed.append(ids)
            return torch.tensor(processed, dtype=torch.long)
        
        src_tensor = process_ids(input_ids)
        tgt_tensor = process_ids(target_ids, add_bos=True, add_eos=True) 
        
        return {
            'src_ids': src_tensor,
            'tgt_ids': tgt_tensor
        }


class NoamLR:
    """Noam learning rate scheduler (from Attention Is All You Need)."""
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.factor * (
            self.model_size ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def save_checkpoint(state, is_best, output_dir):
    """Save model checkpoint."""
    filename = os.path.join(output_dir, "checkpoint.pt")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir, "best_model.pt"))


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler_step' in checkpoint:
        scheduler.step_num = checkpoint['scheduler_step']
    return checkpoint.get('epoch', 0)


def get_optimizer(model, optimizer_name, lr, weight_decay=0.01):
    """Create optimizer based on name."""
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    elif optimizer_name == 'adafactor':
        if not ADAFACTOR_AVAILABLE:
            raise ValueError("Adafactor requires transformers library. Install with: pip install transformers")
        return Adafactor(model.parameters(), lr=lr, relative_step=False, warmup_init=False, scale_parameter=False)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def main():
    parser = argparse.ArgumentParser(description="Train custom transformer for Akkadian-English translation")
    
    # Basic training args
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sp_model", type=str, default="models/tokenizers/spm_akkadian_1000.model")
    parser.add_argument("--output_dir", type=str, default="models/custom_transformer")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--use_alibi", action="store_true", help="Use ALiBi relative position encodings")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    # Phase 6: Optimization args
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "adafactor"],
                        help="Optimizer to use")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.0-0.2)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for model")
    
    # Phase 6: Logging args
    parser.add_argument("--tensorboard_dir", type=str, default="runs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment (for TensorBoard)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    # Setup TensorBoard
    exp_name = args.experiment_name or f"custom_transformer_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, exp_name))
    print(f"TensorBoard logging to: {os.path.join(args.tensorboard_dir, exp_name)}")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    # Dataset
    raw_datasets = get_akkadian_dataset()
    if args.smoke_test:
        raw_datasets["train"] = raw_datasets["train"].select(range(20))
        raw_datasets["validation"] = raw_datasets["validation"].select(range(20))
        
    # Collator/Tokenizer
    collator = SPTokenizerCollator(args.sp_model)
    pad_id = collator.pad_id
    vocab_size = collator.sp.vocab_size()
    
    train_loader = DataLoader(raw_datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(raw_datasets["validation"], batch_size=args.batch_size, collate_fn=collator)
    
    # Log hyperparameters to TensorBoard
    hparams = vars(args)
    writer.add_text("hyperparameters", str(hparams))
    
    # Model with configurable dropout
    model = TransformerModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=256, 
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=args.dropout,
        use_alibi=args.use_alibi
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with optional label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=args.label_smoothing)
    
    # Optimizer based on selection
    optimizer = get_optimizer(model, args.optimizer, lr=0, weight_decay=0.01)
    print(f"Using optimizer: {args.optimizer}")
    
    # Scheduler (Noam)
    scheduler = NoamLR(optimizer, model_size=256, warmup_steps=4000)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume_from:
        start_epoch = load_checkpoint(args.resume_from, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")

    # Early Stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    best_loss = float('inf')
    metrics_log = []
    
    print(f"\nStarting training:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Dropout: {args.dropout}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train(
            model, train_loader, optimizer, criterion, device, 
            pad_token_id=pad_id, scheduler=scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        val_loss = evaluate(model, val_loader, criterion, device, pad_token_id=pad_id)
        
        current_lr = optimizer.param_groups[0]['lr']
        ppl = math.exp(val_loss) if val_loss < 100 else float('inf')
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.4f} | LR: {current_lr:.2e}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        writer.add_scalar('Loss/validation', val_loss, epoch + 1)
        writer.add_scalar('Perplexity', ppl, epoch + 1)
        writer.add_scalar('Learning_Rate', current_lr, epoch + 1)
        
        # JSON metrics logging
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "ppl": ppl,
            "lr": current_lr
        }
        metrics_log.append(metrics)
        
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_log, f, indent=2)

        # Checkpoint
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler_step': scheduler.step_num,
            'args': vars(args)
        }, is_best, args.output_dir)
        
        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    writer.close()
    print(f"\nTraining complete! Best val loss: {best_loss:.4f}")
    print(f"TensorBoard logs: tensorboard --logdir {args.tensorboard_dir}")


if __name__ == "__main__":
    main()
