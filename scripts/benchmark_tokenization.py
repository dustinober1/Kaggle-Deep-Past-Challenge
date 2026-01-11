import sentencepiece as spm
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

def benchmark_tokenization():
    models = {
        'BPE-1000 (Special)': 'models/tokenizers/spm_akkadian_1000_special.model',
        'Character (Baseline)': 'models/tokenizers/spm_akkadian_char.model'
    }
    
    train_path = 'data/processed/train_final.csv'
    if not os.path.exists(train_path):
        print("Dataset not found")
        return

    df = pd.read_csv(train_path)
    samples = df['transliteration'].dropna().astype(str).tolist()
    
    results = []
    
    print(f"Benchmarking on {len(samples)} sentences...")
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Model {name} not found at {path}")
            continue
            
        sp = spm.SentencePieceProcessor(model_file=path)
        
        total_tokens = 0
        total_chars = 0
        unk_count = 0
        
        token_lengths = []
        
        for text in samples:
            tokens = sp.encode(text, out_type=str)
            ids = sp.encode(text, out_type=int)
            
            total_tokens += len(tokens)
            total_chars += len(text)
            token_lengths.append(len(tokens))
            
            # Count unknowns (ID 0 is usually unk in SPM)
            unk_count += ids.count(0)
            
        avg_len = total_tokens / len(samples)
        compression = total_chars / total_tokens if total_tokens > 0 else 0
        
        results.append({
            'Model': name,
            'Avg Tokens/Sent': avg_len,
            'Chars/Token': compression,
            'vocab_size': sp.get_piece_size(),
            'UNK %': (unk_count / total_tokens) * 100 if total_tokens > 0 else 0
        })
        
        print(f"\n{name}:")
        print(f"  Avg Tokens/Sent: {avg_len:.2f}")
        print(f"  Chars/Token: {compression:.2f}")
        print(f"  Sample: {samples[0]}")
        sample_tokens = sp.encode(samples[0], out_type=str)
        print(f"  Tokenized: {sample_tokens[:20]}...")

    df_res = pd.DataFrame(results)
    print("\nBENCHMARK RESULTS")
    print(df_res.to_string(index=False))
    
    # Simple recommendation
    best = df_res.loc[df_res['Chars/Token'].idxmax()]
    print(f"\nRecommendation: {best['Model']} (Efficiency: {best['Chars/Token']:.2f} chars/token)")

if __name__ == "__main__":
    benchmark_tokenization()
