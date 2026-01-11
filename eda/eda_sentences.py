#!/usr/bin/env python3
"""
EDA for Sentences_Oare_FirstWord_LinNum.csv - Sentence alignment helper
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

def load_data():
    """Load sentences alignment file."""
    print("=== SENTENCES_OARE_FIRSTWORD_LINNUM.CSV ===")
    df = pd.read_csv(DATA_DIR / 'Sentences_Oare_FirstWord_LinNum.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def analyze_document_distribution(df):
    """Analyze sentences per document."""
    print("\n=== DOCUMENT DISTRIBUTION ===")
    
    doc_counts = df['text_uuid'].value_counts()
    print(f"Total documents: {len(doc_counts):,}")
    print(f"Total sentences: {len(df):,}")
    print(f"Average sentences per document: {len(df)/len(doc_counts):.1f}")
    
    print(f"\nSentences per document distribution:")
    print(doc_counts.describe().round(1))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(doc_counts.values, bins=30, color='steelblue', alpha=0.7)
    ax.set_xlabel('Sentences per Document')
    ax.set_ylabel('Number of Documents')
    ax.set_title('Distribution of Sentences per Document')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sentences_per_document.png', dpi=150)
    plt.close()
    
    return doc_counts

def analyze_tablet_structure(df):
    """Analyze side and column distribution."""
    print("\n=== TABLET STRUCTURE ===")
    
    if 'side' in df.columns:
        side_counts = df['side'].value_counts()
        print(f"Side distribution:")
        print(side_counts)
    
    if 'column' in df.columns:
        col_counts = df['column'].value_counts()
        print(f"\nColumn distribution:")
        print(col_counts)

def analyze_line_numbers(df):
    """Analyze line number patterns."""
    print("\n=== LINE NUMBER PATTERNS ===")
    
    if 'line_number' in df.columns:
        # Convert to numeric where possible
        line_nums = pd.to_numeric(df['line_number'], errors='coerce')
        valid_nums = line_nums.dropna()
        
        print(f"Valid numeric line numbers: {len(valid_nums):,}")
        print(f"Line number stats:")
        print(valid_nums.describe().round(1))

def analyze_first_words(df):
    """Analyze common first words."""
    print("\n=== FIRST WORD PATTERNS ===")
    
    if 'first_word_spelling' in df.columns:
        first_words = df['first_word_spelling'].dropna()
        word_counts = Counter(first_words)
        print(f"Unique first words: {len(word_counts):,}")
        print(f"Most common first words:")
        for word, count in word_counts.most_common(15):
            print(f"  {word}: {count}")
        
        # Visualization
        top_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_words['Word'], top_words['Count'], color='teal')
        ax.set_xlabel('Count')
        ax.set_title('Top 20 Sentence-Starting Words')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'sentences_first_words.png', dpi=150)
        plt.close()

def analyze_translations(df):
    """Analyze available translations."""
    print("\n=== TRANSLATION COVERAGE ===")
    
    if 'translation' in df.columns:
        has_translation = df['translation'].notna().sum()
        print(f"Sentences with translations: {has_translation:,} ({100*has_translation/len(df):.1f}%)")
        
        # Sample translations
        sample = df[df['translation'].notna()].sample(min(5, has_translation))
        print("\nSample translations:")
        for _, row in sample.iterrows():
            print(f"  {str(row['translation'])[:100]}...")

def check_train_crossref(df):
    """Cross-reference with train.csv."""
    print("\n=== CROSS-REFERENCE WITH TRAIN ===")
    
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    
    # Check if text_uuid matches oare_id
    train_ids = set(train_df['oare_id'])
    sentence_uuids = set(df['text_uuid'].dropna())
    
    overlap = train_ids & sentence_uuids
    print(f"Train oare_ids: {len(train_ids):,}")
    print(f"Sentence text_uuids: {len(sentence_uuids):,}")
    print(f"Overlap: {len(overlap):,}")
    
    if len(overlap) > 0:
        # For overlapping documents, count sentences
        overlap_sentences = df[df['text_uuid'].isin(overlap)]
        print(f"Total sentences for overlapping documents: {len(overlap_sentences):,}")

def main():
    print("=" * 60)
    print("DEEP PAST CHALLENGE - SENTENCES ALIGNMENT EDA")
    print("=" * 60)
    
    df = load_data()
    analyze_document_distribution(df)
    analyze_tablet_structure(df)
    analyze_line_numbers(df)
    analyze_first_words(df)
    analyze_translations(df)
    check_train_crossref(df)
    
    print("\n" + "=" * 60)
    print("Analysis complete. Figures saved to:", FIG_DIR)
    print("=" * 60)

if __name__ == '__main__':
    main()
