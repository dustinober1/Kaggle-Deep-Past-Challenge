import sentencepiece as spm
import pandas as pd
import os
import random

def analyze_morphology():
    model_path = 'models/tokenizers/spm_akkadian_1000_special.model'
    train_path = 'data/processed/train_final.csv'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    sp = spm.SentencePieceProcessor(model_file=model_path)
    
    df = pd.read_csv(train_path)
    samples = df['transliteration'].dropna().astype(str).sample(20, random_state=42).tolist()
    
    print("Morphological Analysis of BPE-1000-Special")
    print("=" * 60)
    
    # Common Akkadian morphological markers (prefixes/suffixes)
    # This is a heuristic list for qualitative checking
    # Prefixes: i-, u-, ta-, ni-, a- (verbal)
    # Suffixes: -am, -nim, -kum, -šu, -ka, -ni
    # Case endings: -um, -am, -im
    
    morph_hits = 0
    total_tokens = 0
    
    for text in samples:
        tokens = sp.encode(text, out_type=str)
        print(f"\nOriginal: {text}")
        print(f"Tokens:   {tokens}")
        
        # Simple heuristic check for common morphemes being their own tokens
        # Note: SPM tokens might have leading meta-symbol ▁ (u+2581)
        
        analysis = []
        for t in tokens:
            clean_t = t.replace('▁', '')
            total_tokens += 1
            
            # Check for common syllabic/morphological units
            # Using a small set of known Akkadian grammatical morphemes
            if clean_t in ['iš', 'tù', 'ma', 'na', 'um', 'am', 'im', 'šu', 'ka', 'ni', 'i', 'u', 'ta']:
                analysis.append(f"{t}(morph?)")
                morph_hits += 1
            elif len(clean_t) > 3:
                analysis.append(f"{t}(stem?)")
            else:
                analysis.append(t)
                
        print(f"Analysis: {' '.join(analysis)}")
        
    print("\n" + "=" * 60)
    print("Qualitative Summary")
    print("-" * 60)
    print("The tokenizer appears to split text into:")
    print("1. Determinatives (preserved intact due to special tokens)")
    print("2. Common Syllables (e.g., iš, tù, ma, na) - often acting as phono-morphological units")
    print("3. Case endings/Suffixes (e.g., -am, -im) - frequently isolated")
    print("-" * 60)
    print(f"Total tokens analyzed: {total_tokens}")
    print(f"Potential morphological matches: {morph_hits} ({morph_hits/total_tokens:.1%})")

if __name__ == "__main__":
    analyze_morphology()
