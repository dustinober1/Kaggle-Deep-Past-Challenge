import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_transformer import TransformerModel
from dataset_loader import get_akkadian_dataset
import sentencepiece as spm
import argparse
import os
import time
import math
from tqdm import tqdm

def train(model, loader, optimizer, criterion, device, pad_token_id, scheduler=None):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        src = batch['src_ids'].to(device)
        tgt = batch['tgt_ids'].to(device)
        
        # Target input (for decoder) is shifted right (exclude last token)
        tgt_input = tgt[:, :-1]
        
        # Target output (for loss) is shifted left (exclude first token)
        tgt_output = tgt[:, 1:]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(src, tgt_input, pad_token_id)
        
        optimizer.zero_grad()
        
        logits = model(
            src, 
            tgt_input, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask, 
            src_padding_mask=src_padding_mask, 
            tgt_padding_mask=tgt_padding_mask
        )
        
        # Reshape for loss
        # logits: [batch, seq_len, vocab_size] -> [batch*seq_len, vocab_size]
        # tgt_output: [batch, seq_len] -> [batch*seq_len]
        
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, pad_token_id):
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
    def __init__(self, sp_model_path, max_len=128):
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.max_len = max_len
        self.pad_id = self.sp.pad_id()
        self.eos_id = self.sp.eos_id()
        self.bos_id = self.sp.bos_id()

    def __call__(self, batch):
        # batch is list of dicts from dataset
        inputs = [item['transliteration_cleaned'] for item in batch]
        targets = [item['translation_normalized'] for item in batch]
        
        # Encode
        input_ids = [self.sp.encode(text, out_type=int) for text in inputs]
        target_ids = [self.sp.encode(text, out_type=int) for text in targets]
        
        # Pad and Truncate
        def process_ids(ids_list, add_bos=False, add_eos=True):
            processed = []
            for ids in ids_list:
                # Truncate
                if len(ids) > self.max_len - (1 if add_bos else 0) - (1 if add_eos else 0):
                    ids = ids[:self.max_len - (1 if add_bos else 0) - (1 if add_eos else 0)]
                
                # Add specials
                if add_bos:
                    ids = [self.bos_id] + ids
                if add_eos:
                    ids = ids + [self.eos_id]
                    
                # Pad
                if len(ids) < self.max_len:
                    ids = ids + [self.pad_id] * (self.max_len - len(ids))
                
                processed.append(ids)
            return torch.tensor(processed, dtype=torch.long)
        
        src_tensor = process_ids(input_ids)
        # Target usually needs BOS for decoder start
        tgt_tensor = process_ids(target_ids, add_bos=True, add_eos=True) 
        
        return {
            'src_ids': src_tensor,
            'tgt_ids': tgt_tensor
        }

import json
import shutil

class NoamLR:
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
    filename = os.path.join(output_dir, "checkpoint.pt")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir, "best_model.pt"))

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler_step' in checkpoint:
        scheduler.step_num = checkpoint['scheduler_step']
    return checkpoint.get('epoch', 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sp_model", type=str, default="models/tokenizers/spm_akkadian_1000.model")
    parser.add_argument("--output_dir", type=str, default="models/custom_transformer")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--use_alibi", action="store_true", help="Use ALiBi relative position encodings")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
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
    
    # Model
    model = TransformerModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=256, 
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        use_alibi=args.use_alibi
    ).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    # Using simple Adam with placeholder LR, scheduler controls it
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    
    # Scheduler
    scheduler = NoamLR(optimizer, model_size=256, warmup_steps=4000)
    
    # Resume
    start_epoch = 0
    if args.resume_from:
        start_epoch = load_checkpoint(args.resume_from, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")

    # Early Stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    best_loss = float('inf')
    metrics_log = []
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device, pad_token_id=pad_id, scheduler=scheduler)
        # Check LR
        current_lr = optimizer.param_groups[0]['lr']
        # Step scheduler each batch usually, but here doing basic step per epoch or batch? 
        # Noam usually per step. Fix: call scheduler.step() inside train loop? 
        # For simplicity in Custom Loop implementation, we'll keep it per epoch or just call it inside train() if needed.
        # Implemented NoamLR.step() as manual call. Let's start by modifying train() to take scheduler?
        # Or just step per epoch for now to avoid complexity in `train` function modification. 
        # Actually Noam MUST be per step. I will hack it: call scheduler.step() in the train loop below.
        
        # Note: I didn't pass scheduler to train(). I should update update train() too or just step it here N times?
        # Better: pass scheduler to train().
        
        val_loss = evaluate(model, val_loader, criterion, device, pad_token_id=pad_id)
        
        ppl = math.exp(val_loss) if val_loss < 100 else float('inf')
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.4f} | LR: {current_lr:.2e}")
        
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "ppl": ppl,
            "lr": current_lr
        }
        metrics_log.append(metrics)
        
        # Save logs
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
            'scheduler_step': scheduler.step_num
        }, is_best, args.output_dir)
        
        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    main()
