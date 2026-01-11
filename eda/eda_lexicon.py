#!/usr/bin/env python3
"""
EDA for lexicon and dictionary files
Deep Past Challenge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
DATA_DIR = Path(__file__).parent.parent / 'data'
FIG_DIR = Path(__file__).parent / 'figures'

def load_lexicon():
    """Load OA_Lexicon_eBL.csv."""
    print("=== OA_LEXICON_EBL.CSV ===")
    df = pd.read_csv(DATA_DIR / 'OA_Lexicon_eBL.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def load_dictionary():
    """Load eBL_Dictionary.csv."""
    print("\n=== EBL_DICTIONARY.CSV ===")
    df = pd.read_csv(DATA_DIR / 'eBL_Dictionary.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def analyze_lexicon_types(lex_df):
    """Analyze word type distribution in lexicon."""
    print("\n=== LEXICON WORD TYPES ===")
    
    type_counts = lex_df['type'].value_counts()
    print(f"Word type distribution:")
    print(type_counts)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    type_counts.head(15).plot(kind='barh', ax=ax, color='teal')
    ax.set_xlabel('Count')
    ax.set_title('Word Type Distribution in Lexicon')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'lexicon_word_types.png', dpi=150)
    plt.close()
    
    return type_counts

def analyze_lexicon_completeness(lex_df):
    """Check field completeness in lexicon."""
    print("\n=== LEXICON FIELD COMPLETENESS ===")
    
    for col in lex_df.columns:
        non_null = lex_df[col].notna().sum()
        pct = 100 * non_null / len(lex_df)
        print(f"  {col}: {non_null:,} ({pct:.1f}%)")

def analyze_dictionary_coverage(dict_df):
    """Analyze dictionary definitions."""
    print("\n=== DICTIONARY COVERAGE ===")
    
    print(f"Total entries: {len(dict_df):,}")
    print(f"Entries with definitions: {dict_df['definition'].notna().sum():,}")
    print(f"Entries with derivation info: {dict_df['derived_from'].notna().sum():,}")
    
    # Definition length distribution
    dict_df['def_length'] = dict_df['definition'].str.len()
    print(f"\nDefinition length statistics:")
    print(dict_df['def_length'].describe().round(0))
    
    # Sample definitions
    print("\n=== SAMPLE DEFINITIONS ===")
    sample = dict_df[dict_df['definition'].notna()].sample(5)
    for _, row in sample.iterrows():
        print(f"  {row['word']}: {str(row['definition'])[:100]}...")

def check_train_vocabulary_coverage(lex_df):
    """Check how much of train vocabulary is in lexicon."""
    print("\n=== TRAIN VOCABULARY COVERAGE ===")
    
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    
    # Get all unique words from training transliterations
    all_words = set()
    for trans in train_df['transliteration']:
        # Split on spaces and hyphens for Akkadian
        words = trans.replace('-', ' ').split()
        all_words.update(w.lower() for w in words)
    
    print(f"Unique tokens in train (after hyphen split): {len(all_words):,}")
    
    # Get lexicon forms
    lex_forms = set(lex_df['form'].str.lower().dropna())
    lex_norms = set(lex_df['norm'].str.lower().dropna())
    
    # Coverage
    form_coverage = len(all_words & lex_forms)
    norm_coverage = len(all_words & lex_norms)
    
    print(f"Covered by lexicon 'form': {form_coverage:,} ({100*form_coverage/len(all_words):.1f}%)")
    print(f"Covered by lexicon 'norm': {norm_coverage:,} ({100*norm_coverage/len(all_words):.1f}%)")
    
    # Uncovered words (sample)
    uncovered = all_words - lex_forms - lex_norms
    print(f"Uncovered tokens: {len(uncovered):,}")
    print(f"Sample uncovered: {list(uncovered)[:20]}")
    
    return all_words, lex_forms, uncovered

def analyze_normalization_patterns(lex_df):
    """Analyze form → norm → lexeme normalization."""
    print("\n=== NORMALIZATION PATTERNS ===")
    
    # Sample normalization chain
    sample = lex_df[['form', 'norm', 'lexeme']].dropna().sample(10)
    print("Sample normalization chains (form → norm → lexeme):")
    for _, row in sample.iterrows():
        print(f"  {row['form']} → {row['norm']} → {row['lexeme']}")
    
    # Check how many forms map to same lexeme
    lexeme_counts = lex_df['lexeme'].value_counts()
    print(f"\nForms per lexeme stats:")
    print(lexeme_counts.describe().round(1))
    print(f"\nLexemes with most forms:")
    print(lexeme_counts.head(10))

def main():
    print("=" * 60)
    print("DEEP PAST CHALLENGE - LEXICON & DICTIONARY EDA")
    print("=" * 60)
    
    lex_df = load_lexicon()
    dict_df = load_dictionary()
    
    analyze_lexicon_types(lex_df)
    analyze_lexicon_completeness(lex_df)
    analyze_normalization_patterns(lex_df)
    check_train_vocabulary_coverage(lex_df)
    
    analyze_dictionary_coverage(dict_df)
    
    print("\n" + "=" * 60)
    print("Analysis complete. Figures saved to:", FIG_DIR)
    print("=" * 60)

if __name__ == '__main__':
    main()
