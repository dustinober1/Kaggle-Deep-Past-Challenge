#!/usr/bin/env python3
"""
EDA for published_texts.csv - Metadata and coverage analysis
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
    """Load published_texts.csv."""
    df = pd.read_csv(DATA_DIR / 'published_texts.csv')
    print("=== PUBLISHED_TEXTS.CSV OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def analyze_metadata_completeness(df):
    """Check which metadata fields are populated."""
    print("\n=== METADATA COMPLETENESS ===")
    
    completeness = {}
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        completeness[col] = {'count': non_null, 'pct': pct}
        print(f"  {col}: {non_null:,} ({pct:.1f}%)")
    
    # Visualization
    comp_df = pd.DataFrame(completeness).T.reset_index()
    comp_df.columns = ['Column', 'Count', 'Percentage']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(comp_df['Column'], comp_df['Percentage'], color='steelblue')
    ax.set_xlabel('Completeness (%)')
    ax.set_title('Metadata Field Completeness')
    ax.set_xlim(0, 100)
    ax.axvline(50, color='red', linestyle='--', alpha=0.5)
    
    for bar, pct in zip(bars, comp_df['Percentage']):
        ax.text(min(pct + 1, 95), bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'published_texts_completeness.png', dpi=150)
    plt.close()
    
    return completeness

def analyze_genre_distribution(df):
    """Analyze genre labels."""
    print("\n=== GENRE DISTRIBUTION ===")
    
    if 'genre_label' in df.columns:
        genres = df['genre_label'].dropna()
        genre_counts = genres.value_counts()
        print(f"Texts with genre labels: {len(genres):,} ({100*len(genres)/len(df):.1f}%)")
        print(f"\nGenre distribution:")
        print(genre_counts)
        
        if len(genre_counts) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            genre_counts.head(15).plot(kind='barh', ax=ax, color='teal')
            ax.set_xlabel('Count')
            ax.set_title('Top 15 Genre Labels')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(FIG_DIR / 'published_texts_genres.png', dpi=150)
            plt.close()
        
        return genre_counts
    else:
        print("No genre_label column found")
        return None

def analyze_external_ids(df):
    """Analyze external ID coverage."""
    print("\n=== EXTERNAL ID COVERAGE ===")
    
    id_cols = ['cdli_id', 'eBL_id', 'oatp_key', 'excavation_no']
    
    for col in id_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null:,} ({100*non_null/len(df):.1f}%)")

def analyze_train_overlap(df):
    """Check overlap with training data."""
    print("\n=== OVERLAP WITH TRAIN.CSV ===")
    
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_ids = set(train_df['oare_id'])
    published_ids = set(df['oare_id'])
    
    overlap = train_ids & published_ids
    only_in_train = train_ids - published_ids
    only_in_published = published_ids - train_ids
    
    print(f"Train samples: {len(train_ids):,}")
    print(f"Published texts: {len(published_ids):,}")
    print(f"Overlap (in both): {len(overlap):,}")
    print(f"Only in train: {len(only_in_train):,}")
    print(f"Only in published (no translation): {len(only_in_published):,}")
    
    return overlap, only_in_published

def analyze_transliteration_stats(df):
    """Analyze transliteration statistics."""
    print("\n=== TRANSLITERATION STATISTICS ===")
    
    if 'transliteration' in df.columns:
        df['text_length'] = df['transliteration'].str.len()
        df['word_count'] = df['transliteration'].str.split().str.len()
        
        print(f"Text length (chars):")
        print(df['text_length'].describe().round(1))
        print(f"\nWord count:")
        print(df['word_count'].describe().round(1))
        
        # Distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(df['text_length'].dropna(), bins=50, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Character Count')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Transliteration Length Distribution')
        
        axes[1].hist(df['word_count'].dropna(), bins=50, color='coral', alpha=0.7)
        axes[1].set_xlabel('Word Count')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Word Count Distribution')
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'published_texts_lengths.png', dpi=150)
        plt.close()

def analyze_notes_commentary(df):
    """Check notes and commentary fields for potential translations."""
    print("\n=== NOTES & COMMENTARY ===")
    
    if 'note' in df.columns:
        notes = df['note'].dropna()
        print(f"Texts with notes: {len(notes):,} ({100*len(notes)/len(df):.1f}%)")
        if len(notes) > 0:
            print(f"Sample notes:")
            for note in notes.head(3):
                print(f"  - {str(note)[:200]}...")
    
    if 'interlinear_commentary' in df.columns:
        comm = df['interlinear_commentary'].dropna()
        print(f"\nTexts with interlinear commentary: {len(comm):,}")

def main():
    print("=" * 60)
    print("DEEP PAST CHALLENGE - PUBLISHED_TEXTS.CSV EDA")
    print("=" * 60)
    
    df = load_data()
    analyze_metadata_completeness(df)
    analyze_genre_distribution(df)
    analyze_external_ids(df)
    analyze_train_overlap(df)
    analyze_transliteration_stats(df)
    analyze_notes_commentary(df)
    
    print("\n" + "=" * 60)
    print("Analysis complete. Figures saved to:", FIG_DIR)
    print("=" * 60)

if __name__ == '__main__':
    main()
