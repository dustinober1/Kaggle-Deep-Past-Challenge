#!/usr/bin/env python3
"""
EDA for train.csv - Core training data analysis
Deep Past Challenge: Translate Akkadian to English
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from pathlib import Path

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
DATA_DIR = Path(__file__).parent.parent / 'data'
FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

def load_data():
    """Load train.csv and basic info."""
    df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"=== TRAIN.CSV OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df

def analyze_text_lengths(df):
    """Analyze character and word counts."""
    print("\n=== TEXT LENGTH ANALYSIS ===")
    
    # Character counts
    df['trans_chars'] = df['transliteration'].str.len()
    df['transl_chars'] = df['translation'].str.len()
    
    # Word counts (simple split)
    df['trans_words'] = df['transliteration'].str.split().str.len()
    df['transl_words'] = df['translation'].str.split().str.len()
    
    # Ratios
    df['char_ratio'] = df['transl_chars'] / df['trans_chars']
    df['word_ratio'] = df['transl_words'] / df['trans_words']
    
    stats = df[['trans_chars', 'transl_chars', 'trans_words', 'transl_words', 
                'char_ratio', 'word_ratio']].describe()
    print("\nStatistics:")
    print(stats.round(2))
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Character distributions
    axes[0, 0].hist(df['trans_chars'], bins=50, alpha=0.7, label='Transliteration', color='steelblue')
    axes[0, 0].hist(df['transl_chars'], bins=50, alpha=0.7, label='Translation', color='coral')
    axes[0, 0].set_xlabel('Character Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Character Count Distribution')
    axes[0, 0].legend()
    
    # Word distributions
    axes[0, 1].hist(df['trans_words'], bins=50, alpha=0.7, label='Transliteration', color='steelblue')
    axes[0, 1].hist(df['transl_words'], bins=50, alpha=0.7, label='Translation', color='coral')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].legend()
    
    # Character ratio
    axes[1, 0].hist(df['char_ratio'], bins=50, color='purple', alpha=0.7)
    axes[1, 0].axvline(df['char_ratio'].median(), color='red', linestyle='--', label=f'Median: {df["char_ratio"].median():.2f}')
    axes[1, 0].set_xlabel('Translation/Transliteration Character Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Character Expansion Ratio')
    axes[1, 0].legend()
    
    # Word ratio
    axes[1, 1].hist(df['word_ratio'], bins=50, color='green', alpha=0.7)
    axes[1, 1].axvline(df['word_ratio'].median(), color='red', linestyle='--', label=f'Median: {df["word_ratio"].median():.2f}')
    axes[1, 1].set_xlabel('Translation/Transliteration Word Ratio')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Word Expansion Ratio')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'train_length_distributions.png', dpi=150)
    plt.close()
    
    return df

def analyze_vocabulary(df):
    """Analyze vocabulary and special patterns."""
    print("\n=== VOCABULARY ANALYSIS ===")
    
    # Transliteration vocabulary
    all_trans_words = ' '.join(df['transliteration']).split()
    trans_vocab = Counter(all_trans_words)
    print(f"\nTransliteration:")
    print(f"  Total tokens: {len(all_trans_words):,}")
    print(f"  Unique tokens: {len(trans_vocab):,}")
    print(f"  Top 20 tokens: {trans_vocab.most_common(20)}")
    
    # Translation vocabulary
    all_transl_words = ' '.join(df['translation']).split()
    transl_vocab = Counter(all_transl_words)
    print(f"\nTranslation:")
    print(f"  Total tokens: {len(all_transl_words):,}")
    print(f"  Unique tokens: {len(transl_vocab):,}")
    print(f"  Top 20 tokens: {transl_vocab.most_common(20)}")
    
    return trans_vocab, transl_vocab

def analyze_akkadian_patterns(df):
    """Analyze Akkadian-specific patterns: determinatives, logograms, etc."""
    print("\n=== AKKADIAN PATTERNS ===")
    
    all_trans = ' '.join(df['transliteration'])
    
    # Determinatives in curly brackets {d}, {ki}, etc.
    determinatives = re.findall(r'\{[^}]+\}', all_trans)
    det_counts = Counter(determinatives)
    print(f"\nDeterminatives found: {len(determinatives):,}")
    print(f"  Unique: {len(det_counts)}")
    print(f"  Top 15: {det_counts.most_common(15)}")
    
    # Sumerian logograms (ALL CAPS words)
    logograms = re.findall(r'\b[A-ZŠṢṬḪ₀-₉\.]+\b', all_trans)
    logo_counts = Counter(logograms)
    print(f"\nLogograms (ALL CAPS): {len(logograms):,}")
    print(f"  Unique: {len(logo_counts)}")
    print(f"  Top 15: {logo_counts.most_common(15)}")
    
    # Proper nouns (Capitalized first letter)
    proper_nouns = re.findall(r'\b[A-ZŠṢṬḪ][a-zšṣṭḫāēīūàèìù₀-₉-]+\b', all_trans)
    proper_counts = Counter(proper_nouns)
    print(f"\nProper nouns (Capitalized): {len(proper_nouns):,}")
    print(f"  Unique: {len(proper_counts)}")
    print(f"  Top 15: {proper_counts.most_common(15)}")
    
    # Hyphenated syllables (Akkadian signature)
    hyphenated = re.findall(r'\b[a-zšṣṭḫāēīūàèìù₀-₉]+-[a-zšṣṭḫāēīūàèìù₀-₉-]+\b', all_trans, re.IGNORECASE)
    print(f"\nHyphenated words: {len(hyphenated):,}")
    
    # Special characters
    special_chars = set(re.findall(r'[šṣṭḫāēīūàèìùŠṢṬḪĀĒĪŪÀÈÌÙ₀₁₂₃₄₅₆₇₈₉]', all_trans))
    print(f"\nSpecial Akkadian characters: {special_chars}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Determinatives bar chart
    det_df = pd.DataFrame(det_counts.most_common(15), columns=['Determinative', 'Count'])
    axes[0].barh(det_df['Determinative'], det_df['Count'], color='teal')
    axes[0].set_xlabel('Count')
    axes[0].set_title('Top 15 Determinatives')
    axes[0].invert_yaxis()
    
    # Logograms bar chart
    logo_df = pd.DataFrame(logo_counts.most_common(15), columns=['Logogram', 'Count'])
    axes[1].barh(logo_df['Logogram'], logo_df['Count'], color='darkorange')
    axes[1].set_xlabel('Count')
    axes[1].set_title('Top 15 Sumerian Logograms')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'train_akkadian_patterns.png', dpi=150)
    plt.close()
    
    return det_counts, logo_counts

def analyze_translation_patterns(df):
    """Analyze English translation patterns."""
    print("\n=== TRANSLATION PATTERNS ===")
    
    all_transl = ' '.join(df['translation'])
    
    # Common phrases
    words = all_transl.lower().split()
    print(f"\nMost common words (excluding punctuation):")
    word_counts = Counter(words)
    print(word_counts.most_common(30))
    
    # Look for translation conventions
    # Brackets for uncertain readings
    brackets = re.findall(r'\[[^\]]+\]', all_transl)
    print(f"\nBracketed text (uncertain/restored): {len(brackets)} occurrences")
    
    # Ellipsis for gaps
    ellipsis = re.findall(r'\.\.\.', all_transl)
    print(f"Ellipsis (...): {len(ellipsis)} occurrences")
    
    # Quoted text
    quotes = re.findall(r'"[^"]+"', all_transl)
    print(f"Quoted text: {len(quotes)} occurrences")
    
    return word_counts

def analyze_test_structure():
    """Quick look at test.csv structure."""
    print("\n=== TEST.CSV STRUCTURE ===")
    df = pd.read_csv(DATA_DIR / 'test.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:")
    print(df.head())
    print(f"\nNote: Real test has ~4000 sentences from ~400 docs")
    return df

def generate_summary(df, trans_vocab, transl_vocab):
    """Generate summary statistics for report."""
    summary = {
        'total_samples': len(df),
        'avg_trans_chars': df['trans_chars'].mean(),
        'avg_transl_chars': df['transl_chars'].mean(),
        'avg_trans_words': df['trans_words'].mean(),
        'avg_transl_words': df['transl_words'].mean(),
        'median_char_ratio': df['char_ratio'].median(),
        'median_word_ratio': df['word_ratio'].median(),
        'trans_vocab_size': len(trans_vocab),
        'transl_vocab_size': len(transl_vocab),
    }
    
    print("\n=== SUMMARY FOR REPORT ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:,}")
    
    return summary

def main():
    print("=" * 60)
    print("DEEP PAST CHALLENGE - TRAIN.CSV EDA")
    print("=" * 60)
    
    df = load_data()
    df = analyze_text_lengths(df)
    trans_vocab, transl_vocab = analyze_vocabulary(df)
    analyze_akkadian_patterns(df)
    analyze_translation_patterns(df)
    analyze_test_structure()
    summary = generate_summary(df, trans_vocab, transl_vocab)
    
    print("\n" + "=" * 60)
    print("Figures saved to:", FIG_DIR)
    print("=" * 60)
    
    return summary

if __name__ == '__main__':
    main()
