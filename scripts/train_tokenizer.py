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

    # Save to temp file
    with open(temp_text_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')

    # Train models
    # Based on experiment, max BPE vocab is ~2400.
    vocab_sizes = [500, 1000, 2000]

    for vocab_size in vocab_sizes:

        model_prefix = os.path.join(model_dir, f'spm_akkadian_{vocab_size}')
        print(f"Training tokenizer with vocab size {vocab_size}...")
        
        # SentencePiece arguments
        # input: input file
        # model_prefix: output model name prefix
        # vocab_size: vocabulary size
        # character_coverage: 1.0 ensures all characters are covered (good for languages with specific chars)
        # model_type: 'bpe' or 'unigram' (default). unigram is often better for agglutinative languages.
        
        spm.SentencePieceTrainer.train(
            input=temp_text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type='bpe', # Switch to BPE
            input_sentence_size=1000000,
            shuffle_input_sentence=True
        )
        print(f"Finished training {model_prefix}")
        print(f"Finished training {model_prefix}")

    # Clean up
    if os.path.exists(temp_text_file):
        os.remove(temp_text_file)
    
    print("All tokenizers trained successfully.")

if __name__ == "__main__":
    train_tokenizer()
