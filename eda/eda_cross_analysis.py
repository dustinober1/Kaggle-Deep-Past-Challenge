#!/usr/bin/env python3
"""
Cross-file analysis and relationship mapping
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

def load_all_data():
    """Load key datasets."""
    print("=== LOADING DATASETS ===")
    
    data = {}
    data['train'] = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"train.csv: {len(data['train']):,} rows")
    
    data['published_texts'] = pd.read_csv(DATA_DIR / 'published_texts.csv')
    print(f"published_texts.csv: {len(data['published_texts']):,} rows")
    
    data['sentences'] = pd.read_csv(DATA_DIR / 'Sentences_Oare_FirstWord_LinNum.csv')
    print(f"sentences.csv: {len(data['sentences']):,} rows")
    
    data['bibliography'] = pd.read_csv(DATA_DIR / 'bibliography.csv')
    print(f"bibliography.csv: {len(data['bibliography']):,} rows")
    
    data['lexicon'] = pd.read_csv(DATA_DIR / 'OA_Lexicon_eBL.csv')
    print(f"lexicon.csv: {len(data['lexicon']):,} rows")
    
    return data

def analyze_train_published_join(data):
    """Join train with published_texts to enrich metadata."""
    print("\n=== TRAIN ↔ PUBLISHED_TEXTS JOIN ===")
    
    train = data['train']
    published = data['published_texts']
    
    # Join on oare_id
    enriched = train.merge(published, on='oare_id', how='left', suffixes=('_train', '_pub'))
    
    print(f"Train samples: {len(train):,}")
    print(f"Successfully joined: {enriched['label'].notna().sum():,}")
    
    # Check which metadata is available for train
    metadata_cols = ['genre_label', 'description', 'cdli_id', 'eBL_id', 'note']
    for col in metadata_cols:
        if col in enriched.columns:
            available = enriched[col].notna().sum()
            print(f"  {col}: {available} ({100*available/len(train):.1f}%)")
    
    # Genre distribution for training data
    if 'genre_label' in enriched.columns:
        genres = enriched['genre_label'].value_counts()
        print(f"\nTrain genre distribution:")
        print(genres)
        
        if len(genres) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            genres.head(10).plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Count')
            ax.set_title('Genre Distribution in Training Data')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(FIG_DIR / 'cross_train_genres.png', dpi=150)
            plt.close()
    
    return enriched

def analyze_untranslated_texts(data):
    """Identify texts without translations."""
    print("\n=== UNTRANSLATED TEXTS ANALYSIS ===")
    
    train_ids = set(data['train']['oare_id'])
    published_ids = set(data['published_texts']['oare_id'])
    
    untranslated = published_ids - train_ids
    print(f"Published texts: {len(published_ids):,}")
    print(f"With translations (train): {len(train_ids):,}")
    print(f"Without translations: {len(untranslated):,}")
    
    # Check metadata for untranslated texts
    untranslated_df = data['published_texts'][data['published_texts']['oare_id'].isin(untranslated)]
    
    if 'genre_label' in untranslated_df.columns:
        genres = untranslated_df['genre_label'].value_counts()
        print(f"\nUntranslated texts by genre:")
        print(genres.head(10))

def analyze_bibliography_coverage(data):
    """Analyze bibliography and year distribution."""
    print("\n=== BIBLIOGRAPHY ANALYSIS ===")
    
    bib = data['bibliography']
    
    # Year distribution
    if 'year' in bib.columns:
        years = pd.to_numeric(bib['year'], errors='coerce').dropna()
        print(f"Publication years:")
        print(years.describe().round(0))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(years, bins=30, color='coral', alpha=0.7)
        ax.set_xlabel('Publication Year')
        ax.set_ylabel('Number of Publications')
        ax.set_title('Distribution of Publication Years')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'cross_publication_years.png', dpi=150)
        plt.close()
    
    # Journal distribution
    if 'journal' in bib.columns:
        journals = bib['journal'].value_counts()
        print(f"\nTop journals/sources:")
        print(journals.head(10))

def estimate_additional_data(data):
    """Estimate potential for additional training data."""
    print("\n=== ADDITIONAL TRAINING DATA POTENTIAL ===")
    
    # From publications.csv - not loaded here due to size, using estimates
    print("From publications.csv:")
    print("  - ~880 PDFs with ~217K pages of OCR text")
    print("  - Potential translations in German, French, English")
    print("  - Requires: ID matching, language translation, alignment")
    
    # From sentence alignment
    sentences = data['sentences']
    sentences_with_trans = sentences['translation'].notna().sum()
    print(f"\nFrom sentence alignment file:")
    print(f"  - Total sentences: {len(sentences):,}")
    print(f"  - With translations: {sentences_with_trans:,}")
    
    # From notes in published_texts
    published = data['published_texts']
    if 'note' in published.columns:
        notes = published['note'].notna().sum()
        print(f"\nFrom published_texts notes:")
        print(f"  - Texts with notes: {notes:,}")
        print("  - May contain partial translations or commentary")

def create_data_linkage_diagram(data):
    """Create summary of data relationships."""
    print("\n=== DATA LINKAGE SUMMARY ===")
    
    linkages = """
    Data Linkage Map:
    
    train.csv ──[oare_id]──► published_texts.csv (enriches with metadata)
         │
         └──[oare_id]──► Sentences.csv (sentence-level alignment)
    
    published_texts.csv ──[cdli_id]──► CDLI website
                       ──[eBL_id]──► eBL website
                       ──[aliases]──► publications.csv (text matching)
    
    publications.csv ──[pdf_name]──► bibliography.csv (source metadata)
    
    OA_Lexicon.csv ──[eBL URL]──► eBL_Dictionary.csv
    
    Key Statistics:
    - Train: 1,561 samples (document-level translations)
    - Published: 7,991 texts (transliterations only)
    - Potential additional: ~6,430 texts if translations extracted from PDFs
    """
    print(linkages)

def main():
    print("=" * 60)
    print("DEEP PAST CHALLENGE - CROSS-FILE ANALYSIS")
    print("=" * 60)
    
    data = load_all_data()
    
    analyze_train_published_join(data)
    analyze_untranslated_texts(data)
    analyze_bibliography_coverage(data)
    estimate_additional_data(data)
    create_data_linkage_diagram(data)
    
    print("\n" + "=" * 60)
    print("Analysis complete. Figures saved to:", FIG_DIR)
    print("=" * 60)

if __name__ == '__main__':
    main()
