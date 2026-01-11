import pandas as pd
import sentencepiece as spm
import os
import glob

def train_tokenizer():
    # Paths
    train_path = 'data/processed/train_final.csv'
    published_path = 'data/published_texts.csv'
    model_dir = 'models/tokenizers'
    temp_text_file = 'temp_training_data.txt'

    os.makedirs(model_dir, exist_ok=True)

    print("Loading data...")
    # Load data
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        train_texts = df_train['transliteration'].dropna().astype(str).tolist()
    else:
        print(f"Warning: {train_path} not found.")
        train_texts = []

    if os.path.exists(published_path):
        df_pub = pd.read_csv(published_path)
        pub_texts = df_pub['transliteration'].dropna().astype(str).tolist()
    else:
        print(f"Warning: {published_path} not found.")
        pub_texts = []

    all_texts = train_texts + pub_texts
    print(f"Total texts for training: {len(all_texts)}")

    # Debug: Check unique whitespace tokens
    unique_tokens = set()
    for text in all_texts:
        unique_tokens.update(text.split())
    print(f"Unique whitespace-separated tokens: {len(unique_tokens)}")

    # Extract special tokens and determinatives
    print("Extracting special tokens and determinatives...")
    special_tokens = set()
    determinatives = set()
    
    import re
    for text in all_texts:
        # Find tags like <gap>, <big_gap>, <PN>, <GN>, <unk>, <s>, </s>
        # Note: <s>, </s>, <unk>, <pad> are handled by SentencePiece automatically, 
        # but we should ensure others like <gap> are treated as user symbols.
        tags = re.findall(r'<[^>]+>', text)
        for tag in tags:
            # Skip HTML-like tags if any, but our cleaning should have left valid ones
            special_tokens.add(tag)
            
        # Find determinatives like {d}, {ki}, {m}, {f}
        # Restrict to short sequences without spaces to avoid capturing long blocks
        dets = re.findall(r'\{[a-zA-Z0-9\.]+\}', text)
        for det in dets:
            determinatives.add(det)
            
    # Combine user defined symbols
    # Exclude standard tokens if they are in the list (SentencePiece handles them)
    reserved = {'<unk>', '<s>', '</s>', '<pad>'}
    user_symbols = sorted(list((special_tokens | determinatives) - reserved))
    
    print(f"Found {len(user_symbols)} user defined symbols: {user_symbols[:10]}...")

    # Save to temp file
    with open(temp_text_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')

    # Train models
    # Based on experiment, max BPE vocab is ~2400.
    vocab_sizes = [1000] # Focusing on the selected model size

    for vocab_size in vocab_sizes:
        model_prefix = os.path.join(model_dir, f'spm_akkadian_{vocab_size}_special')
        print(f"Training tokenizer with vocab size {vocab_size} (with special tokens)...")
        
        spm.SentencePieceTrainer.train(
            input=temp_text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type='bpe', 
            input_sentence_size=1000000,
            shuffle_input_sentence=True,
            user_defined_symbols=user_symbols
        )
        print(f"Finished training {model_prefix}")
        
    # Also train a character-level model for comparison (vocab_size needs to be small enough)
    # Character level treated as BPE with limited operations or just byte fallback?
    # Actually, BPE with vocab_size = char count is effectively char level.
    # Let's count characters.
    unique_chars = set(''.join(all_texts))
    char_vocab_size = len(unique_chars) + len(user_symbols) + 10 # +10 for safety/control
    
    print(f"Training character-level baseline (vocab ~{char_vocab_size})...")
    char_model_prefix = os.path.join(model_dir, 'spm_akkadian_char')
    spm.SentencePieceTrainer.train(
        input=temp_text_file,
        model_prefix=char_model_prefix,
        model_type='char', # Explicit char mode
        vocab_size=max(100, char_vocab_size), # Minimal size
        character_coverage=1.0,
        user_defined_symbols=user_symbols # Still preserve special tokens
    )
    print("Finished training character-level model")

    # Clean up
    if os.path.exists(temp_text_file):
        os.remove(temp_text_file)
    
    print("All tokenizers trained successfully.")

if __name__ == "__main__":
    train_tokenizer()
