#!/usr/bin/env python3
"""
EDA for publications.csv - OCR scholarly publications analysis
Deep Past Challenge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
DATA_DIR = Path(__file__).parent.parent / 'data'
FIG_DIR = Path(__file__).parent / 'figures'

def load_data():
    """Load publications.csv (large file, may take time)."""
    print("=== PUBLICATIONS.CSV OVERVIEW ===")
    print("Loading large file (~580MB)...")
    df = pd.read_csv(DATA_DIR / 'publications.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df

def analyze_pdf_distribution(df):
    """Analyze distribution across PDFs."""
    print("\n=== PDF DISTRIBUTION ===")
    
    pdf_counts = df['pdf_name'].value_counts()
    print(f"Total unique PDFs: {len(pdf_counts):,}")
    print(f"Total pages: {len(df):,}")
    print(f"Average pages per PDF: {len(df)/len(pdf_counts):.1f}")
    print(f"\nPage count distribution:")
    print(pdf_counts.describe().round(1))
    
    # Top PDFs by page count
    print(f"\nTop 10 PDFs by page count:")
    print(pdf_counts.head(10))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(pdf_counts.values, bins=50, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Pages per PDF')
    axes[0].set_ylabel('Number of PDFs')
    axes[0].set_title('PDF Size Distribution')
    
    # Log scale for better visibility
    axes[1].hist(pdf_counts.values, bins=50, color='coral', alpha=0.7, log=True)
    axes[1].set_xlabel('Pages per PDF')
    axes[1].set_ylabel('Number of PDFs (log scale)')
    axes[1].set_title('PDF Size Distribution (Log Scale)')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'publications_pdf_distribution.png', dpi=150)
    plt.close()
    
    return pdf_counts

def analyze_akkadian_content(df):
    """Analyze distribution of Akkadian content."""
    print("\n=== AKKADIAN CONTENT DISTRIBUTION ===")
    
    akkadian_counts = df['has_akkadian'].value_counts()
    print(f"Pages with Akkadian: {akkadian_counts.get(True, 0):,} ({100*akkadian_counts.get(True, 0)/len(df):.1f}%)")
    print(f"Pages without Akkadian: {akkadian_counts.get(False, 0):,}")
    
    # Per-PDF Akkadian density
    pdf_akkadian = df.groupby('pdf_name')['has_akkadian'].agg(['sum', 'count'])
    pdf_akkadian['density'] = pdf_akkadian['sum'] / pdf_akkadian['count']
    
    print(f"\nAkkadian density per PDF:")
    print(pdf_akkadian['density'].describe().round(2))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pdf_akkadian['density'], bins=50, color='teal', alpha=0.7)
    ax.set_xlabel('Akkadian Page Density')
    ax.set_ylabel('Number of PDFs')
    ax.set_title('Distribution of Akkadian Content Density per PDF')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'publications_akkadian_density.png', dpi=150)
    plt.close()
    
    return pdf_akkadian

def sample_text_quality(df, n=10):
    """Sample OCR text quality."""
    print("\n=== OCR TEXT QUALITY SAMPLE ===")
    
    # Sample pages with Akkadian
    akkadian_pages = df[df['has_akkadian'] == True].sample(min(n, len(df[df['has_akkadian'] == True])))
    
    for _, row in akkadian_pages.head(3).iterrows():
        print(f"\n--- {row['pdf_name']} (page {row['page']}) ---")
        text = str(row['page_text'])[:500] if pd.notna(row['page_text']) else "[No text]"
        print(text)
        print("...")

def detect_languages(df, sample_size=1000):
    """Detect language patterns in OCR text."""
    print("\n=== LANGUAGE DETECTION (Heuristic) ===")
    
    sample = df.sample(min(sample_size, len(df)))
    
    # Simple keyword-based detection
    german_keywords = ['und', 'der', 'die', 'das', 'ist', 'von', 'mit', 'für']
    french_keywords = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'est', 'que']
    english_keywords = ['the', 'and', 'is', 'of', 'to', 'in', 'that', 'with']
    
    lang_counts = {'english': 0, 'german': 0, 'french': 0, 'unknown': 0}
    
    for text in sample['page_text'].dropna():
        text_lower = text.lower()
        words = set(text_lower.split())
        
        en_score = len(words & set(english_keywords))
        de_score = len(words & set(german_keywords))
        fr_score = len(words & set(french_keywords))
        
        max_score = max(en_score, de_score, fr_score)
        if max_score == 0:
            lang_counts['unknown'] += 1
        elif en_score == max_score:
            lang_counts['english'] += 1
        elif de_score == max_score:
            lang_counts['german'] += 1
        else:
            lang_counts['french'] += 1
    
    print(f"Language distribution (sample of {len(sample)}):")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count} ({100*count/len(sample):.1f}%)")
    
    return lang_counts

def find_translation_patterns(df, sample_size=5000):
    """Look for translation-indicating patterns in text."""
    print("\n=== TRANSLATION PATTERNS ===")
    
    sample = df[df['has_akkadian'] == True].sample(min(sample_size, len(df[df['has_akkadian'] == True])))
    
    translation_keywords = [
        'translation', 'translat', 'rendered', 'meaning', 
        'translates to', 'translate as', 'übersetzung', 'traduction'
    ]
    
    pattern_counts = {}
    for kw in translation_keywords:
        count = sample['page_text'].str.contains(kw, case=False, na=False).sum()
        pattern_counts[kw] = count
        print(f"  Pages containing '{kw}': {count}")
    
    return pattern_counts

def analyze_text_statistics(df):
    """Analyze text length and content statistics."""
    print("\n=== TEXT STATISTICS ===")
    
    df['text_length'] = df['page_text'].str.len()
    
    print(f"Text length (characters per page):")
    print(df['text_length'].describe().round(0))
    
    # Check for empty or very short pages
    empty_pages = (df['text_length'] < 100).sum()
    print(f"\nVery short pages (<100 chars): {empty_pages:,} ({100*empty_pages/len(df):.1f}%)")

def main():
    print("=" * 60)
    print("DEEP PAST CHALLENGE - PUBLICATIONS.CSV EDA")
    print("=" * 60)
    
    df = load_data()
    analyze_pdf_distribution(df)
    analyze_akkadian_content(df)
    analyze_text_statistics(df)
    detect_languages(df)
    find_translation_patterns(df)
    sample_text_quality(df)
    
    print("\n" + "=" * 60)
    print("Analysis complete. Figures saved to:", FIG_DIR)
    print("=" * 60)

if __name__ == '__main__':
    main()
