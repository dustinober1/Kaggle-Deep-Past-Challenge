import sentencepiece as spm
import pandas as pd
import glob
import os

def evaluate_tokenizers():
    models = sorted(glob.glob('models/tokenizers/*.model'))
    train_path = 'data/processed/train_final.csv'
    
    if not os.path.exists(train_path):
        print("Training data not found.")
        return

    df = pd.read_csv(train_path)
    # Use a sample for evaluation
    sample_texts = df['transliteration'].dropna().astype(str).sample(100, random_state=42).tolist()
    
    results = []
    
    print(f"Evaluating {len(models)} models on {len(sample_texts)} sample sentences...")
    print("-" * 60)

    for model_path in models:
        sp = spm.SentencePieceProcessor(model_file=model_path)
        vocab_size = sp.get_piece_size()
        
        total_tokens = 0
        total_chars = 0
        
        for text in sample_texts:
            tokens = sp.encode(text, out_type=str)
            total_tokens += len(tokens)
            total_chars += len(text)
            
        avg_tokens = total_tokens / len(sample_texts)
        chars_per_token = total_chars / total_tokens
        
        results.append({
            'Model': os.path.basename(model_path),
            'Vocab': vocab_size,
            'Avg Tokens/Sent': round(avg_tokens, 2),
            'Chars/Token': round(chars_per_token, 2)
        })

        # Print a sample tokenization
        print(f"\nModel: {os.path.basename(model_path)} (Vocab: {vocab_size})")
        print(f"Metrics: Avg Tokens/Sent: {avg_tokens:.2f}, Chars/Token: {chars_per_token:.2f}")
        sample_idx = 0
        print(f"Sample: {sample_texts[sample_idx]}")
        print(f"Tokens: {sp.encode(sample_texts[sample_idx], out_type=str)}")

    print("\n" + "-" * 60)
    print("Summary Report")
    print("-" * 60)
    df_res = pd.DataFrame(results).sort_values('Vocab')
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    evaluate_tokenizers()
