#!/usr/bin/env python3
"""
Align and Filter - Phase 1.2 Step 4

Joins extracted translations with published_texts transliterations,
applies quality filters, and creates the final augmented training dataset.

Input: translated_content.csv, published_texts.csv, publication_links.csv
Output: data/processed/publication_extracted_train.csv
"""

import argparse
import re
from pathlib import Path
from typing import Tuple

import pandas as pd


def calculate_length_ratio(translit: str, transl: str) -> float:
    """Calculate word count ratio between transliteration and translation."""
    if pd.isna(translit) or pd.isna(transl):
        return 0.0
    translit_words = len(str(translit).split())
    transl_words = len(str(transl).split())
    if translit_words == 0:
        return 0.0
    return transl_words / translit_words


def has_valid_characters(text: str) -> bool:
    """Check if text has mostly valid Latin characters."""
    if pd.isna(text) or not text:
        return False
    text = str(text)
    # Count ASCII letters + common diacritics
    valid_chars = sum(1 for c in text if c.isalpha() or c in ' .,!?;:\'"()-')
    return valid_chars / len(text) > 0.8 if len(text) > 0 else False


def filter_quality(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Apply quality filters to translation pairs."""
    stats = {
        'input': len(df),
        'has_translit': 0,
        'has_transl': 0,
        'valid_ratio': 0,
        'valid_chars': 0,
        'min_length': 0,
        'final': 0
    }
    
    # Filter 1: Must have transliteration
    df = df[df['transliteration'].notna() & (df['transliteration'].str.len() > 10)]
    stats['has_translit'] = len(df)
    if verbose:
        print(f"After transliteration filter: {len(df)}")
    
    # Filter 2: Must have translation
    df = df[df['translation'].notna() & (df['translation'].str.len() > 10)]
    stats['has_transl'] = len(df)
    if verbose:
        print(f"After translation filter: {len(df)}")
    
    # Filter 3: Length ratio (should be reasonable)
    df['length_ratio'] = df.apply(
        lambda r: calculate_length_ratio(r['transliteration'], r['translation']), 
        axis=1
    )
    df = df[(df['length_ratio'] > 0.3) & (df['length_ratio'] < 5.0)]
    stats['valid_ratio'] = len(df)
    if verbose:
        print(f"After length ratio filter (0.3-5.0): {len(df)}")
    
    # Filter 4: Valid characters in translation
    df = df[df['translation'].apply(has_valid_characters)]
    stats['valid_chars'] = len(df)
    if verbose:
        print(f"After valid characters filter: {len(df)}")
    
    # Filter 5: Minimum content length
    df = df[
        (df['transliteration'].str.split().str.len() >= 5) &
        (df['translation'].str.split().str.len() >= 5)
    ]
    stats['min_length'] = len(df)
    stats['final'] = len(df)
    if verbose:
        print(f"After minimum length filter (5 words): {len(df)}")
    
    return df, stats


def align_with_published_texts(
    translations_df: pd.DataFrame,
    links_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """Align extracted translations with published_texts transliterations."""
    
    # Get oare_ids from links that correspond to each extraction's PDF
    aligned = []
    
    for _, row in translations_df.iterrows():
        pdf_name = row['pdf_name']
        linked_ids = row.get('linked_oare_ids', '')
        
        if pd.isna(linked_ids) or not linked_ids:
            continue
        
        # Each extraction could match multiple texts linked to that PDF
        for oare_id in str(linked_ids).split(','):
            oare_id = oare_id.strip()
            
            # Get transliteration from published_texts
            text_row = texts_df[texts_df['oare_id'] == oare_id]
            if len(text_row) == 0:
                continue
            
            text_row = text_row.iloc[0]
            translit = text_row.get('transliteration')
            
            if pd.isna(translit):
                continue
            
            aligned.append({
                'oare_id': oare_id,
                'transliteration': translit,
                'translation': row['translation'],
                'source_pdf': pdf_name,
                'source_page': row.get('page'),
                'extraction_confidence': row.get('confidence'),
                'original_language': row.get('original_language', 'EN')
            })
    
    df = pd.DataFrame(aligned)
    
    # Remove duplicates with same oare_id
    if len(df) > 0:
        df = df.drop_duplicates(subset=['oare_id'])
    
    # Exclude texts already in train.csv
    if len(df) > 0 and len(train_df) > 0:
        train_ids = set(train_df['oare_id'])
        df = df[~df['oare_id'].isin(train_ids)]
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Align and filter extracted translations')
    parser.add_argument('--input', type=str, default='data/processed/translated_content.csv')
    parser.add_argument('--output', type=str, default='data/processed/publication_extracted_train.csv')
    parser.add_argument('--validate', action='store_true', help='Run validation checks')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    transl_path = base_dir / args.input
    links_path = base_dir / 'data' / 'processed' / 'publication_links.csv'
    texts_path = base_dir / 'data' / 'published_texts.csv'
    train_path = base_dir / 'data' / 'train.csv'
    output_path = base_dir / args.output
    
    # Load data
    print("Loading data...")
    if not transl_path.exists():
        print(f"Error: Run translation_pipeline.py first to create {transl_path}")
        return
    
    translations_df = pd.read_csv(transl_path)
    links_df = pd.read_csv(links_path)
    texts_df = pd.read_csv(texts_path)
    train_df = pd.read_csv(train_path)
    
    print(f"Translated content: {len(translations_df)}")
    print(f"Publication links: {len(links_df)}")
    print(f"Published texts: {len(texts_df)}")
    print(f"Existing training: {len(train_df)}")
    
    # Align
    print("\nAligning with published_texts...")
    aligned_df = align_with_published_texts(
        translations_df, links_df, texts_df, train_df
    )
    print(f"Aligned pairs: {len(aligned_df)}")
    
    # Quality filter
    print("\nApplying quality filters...")
    filtered_df, stats = filter_quality(aligned_df)
    
    # Format for training
    output_df = filtered_df[['oare_id', 'transliteration', 'translation']].copy()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(output_df)} augmented training samples to {output_path}")
    
    if args.validate:
        print("\n=== VALIDATION ===")
        # Check schema matches train.csv
        assert set(output_df.columns) == set(train_df.columns), "Schema mismatch!"
        print("✓ Schema matches train.csv")
        
        # Check no overlap with train.csv
        overlap = set(output_df['oare_id']) & set(train_df['oare_id'])
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping IDs!"
        print("✓ No overlap with train.csv")
        
        # Sample outputs
        print("\n=== SAMPLE OUTPUTS ===")
        for _, row in output_df.head(3).iterrows():
            print(f"\noare_id: {row['oare_id']}")
            print(f"transliteration: {row['transliteration'][:100]}...")
            print(f"translation: {row['translation'][:100]}...")


if __name__ == '__main__':
    main()
