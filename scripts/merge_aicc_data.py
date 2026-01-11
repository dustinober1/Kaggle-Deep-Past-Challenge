#!/usr/bin/env python3
"""
Merge AICC translations with existing training data.

Combines:
- data/train.csv (original 1,561 labeled translations)
- data/processed/aicc_translations.csv (4,755 new translations from AICC)

Output: data/processed/merged_training.csv
"""

import pandas as pd
from pathlib import Path


def main():
    base_dir = Path(__file__).parent.parent
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(base_dir / 'data' / 'train.csv')
    aicc_df = pd.read_csv(base_dir / 'data' / 'processed' / 'aicc_translations.csv')
    
    print(f"Original train.csv: {len(train_df)} rows")
    print(f"AICC translations: {len(aicc_df)} rows")
    
    # Prepare AICC data for merging
    # Rename columns to match train.csv format
    aicc_prepared = aicc_df.rename(columns={
        'existing_transliteration': 'transliteration',
        'aicc_translation': 'translation'
    })[['oare_id', 'transliteration', 'translation']].copy()
    
    # Filter out rows with empty translations
    aicc_prepared = aicc_prepared[aicc_prepared['translation'].str.len() > 0]
    print(f"AICC with translations: {len(aicc_prepared)} rows")
    
    # Check for overlap (should be none since scraper excluded train IDs)
    overlap = set(train_df['oare_id']) & set(aicc_prepared['oare_id'])
    print(f"Overlap with train.csv: {len(overlap)} rows")
    
    # Add source column to track data origin
    train_df['source'] = 'train'
    aicc_prepared['source'] = 'aicc'
    
    # Merge datasets
    merged_df = pd.concat([train_df, aicc_prepared], ignore_index=True)
    
    print(f"\n=== MERGED DATASET ===")
    print(f"Total rows: {len(merged_df)}")
    print(f"From train.csv: {(merged_df['source'] == 'train').sum()}")
    print(f"From AICC: {(merged_df['source'] == 'aicc').sum()}")
    
    # Statistics
    print(f"\nTransliteration stats:")
    print(f"  Avg length: {merged_df['transliteration'].str.len().mean():.1f} chars")
    print(f"  Min length: {merged_df['transliteration'].str.len().min()} chars")
    print(f"  Max length: {merged_df['transliteration'].str.len().max()} chars")
    
    print(f"\nTranslation stats:")
    print(f"  Avg length: {merged_df['translation'].str.len().mean():.1f} chars")
    print(f"  Min length: {merged_df['translation'].str.len().min()} chars")
    print(f"  Max length: {merged_df['translation'].str.len().max()} chars")
    
    # Save merged dataset
    output_path = base_dir / 'data' / 'processed' / 'merged_training.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Also save a version without source column for direct model training
    train_format = merged_df[['oare_id', 'transliteration', 'translation']]
    train_format_path = base_dir / 'data' / 'processed' / 'augmented_train.csv'
    train_format.to_csv(train_format_path, index=False)
    print(f"Saved training format to {train_format_path}")
    
    # Sample output
    print("\n=== SAMPLE NEW ENTRIES ===")
    for _, row in aicc_prepared.head(2).iterrows():
        print(f"\noare_id: {row['oare_id']}")
        print(f"Transliteration: {row['transliteration'][:100]}...")
        print(f"Translation: {row['translation'][:100]}...")


if __name__ == '__main__':
    main()
