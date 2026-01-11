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

def train(model, loader, optimizer, criterion, device, pad_token_id):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sp_model", type=str, default="models/tokenizers/spm_akkadian_1000.model")
    parser.add_argument("--output_dir", type=str, default="models/custom_transformer")
    parser.add_argument("--smoke_test", action="store_true")
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
        d_model=256, # Smaller for baseline/test
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    # Scheduler: Noam ? For now simple or None
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device, pad_token_id=pad_id)
        val_loss = evaluate(model, val_loader, criterion, device, pad_token_id=pad_id)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print("Saved best model.")

if __name__ == "__main__":
    main()
