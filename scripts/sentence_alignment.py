#!/usr/bin/env python3
"""
Sentence Alignment Integration Script
Phase 1, Task 1.1 - Deep Past Challenge

This script:
1. Parses Sentences_Oare_FirstWord_LinNum.csv for sentence-level pairs
2. Joins with train.csv via text_uuid ↔ oare_id
3. Creates a sentence-aligned training dataset
4. Validates alignment quality on a sample
"""

import pandas as pd
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

def load_data():
    """Load the source CSV files."""
    print("=" * 60)
    print("STEP 1: Loading Data Files")
    print("=" * 60)
    
    # Load sentences data
    sentences_path = DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv"
    sentences_df = pd.read_csv(sentences_path)
    print(f"✓ Loaded Sentences: {len(sentences_df):,} rows")
    print(f"  Columns: {list(sentences_df.columns)}")
    
    # Load training data
    train_path = DATA_DIR / "train.csv"
    train_df = pd.read_csv(train_path)
    print(f"✓ Loaded Train: {len(train_df):,} rows")
    print(f"  Columns: {list(train_df.columns)}")
    
    return sentences_df, train_df


def analyze_overlap(sentences_df, train_df):
    """Analyze the overlap between datasets."""
    print("\n" + "=" * 60)
    print("STEP 2: Analyzing UUID Overlap")
    print("=" * 60)
    
    # Get unique text_uuids from sentences
    sentence_uuids = set(sentences_df['text_uuid'].unique())
    train_uuids = set(train_df['oare_id'].unique())
    
    overlap = sentence_uuids & train_uuids
    sentences_only = sentence_uuids - train_uuids
    train_only = train_uuids - sentence_uuids
    
    print(f"Unique UUIDs in Sentences: {len(sentence_uuids):,}")
    print(f"Unique UUIDs in Train: {len(train_uuids):,}")
    print(f"Overlapping UUIDs: {len(overlap):,}")
    print(f"In Sentences only: {len(sentences_only):,}")
    print(f"In Train only: {len(train_only):,}")
    
    return overlap


def create_sentence_aligned_dataset(sentences_df, train_df, overlap):
    """Create the sentence-aligned training dataset."""
    print("\n" + "=" * 60)
    print("STEP 3: Creating Sentence-Aligned Dataset")
    print("=" * 60)
    
    # Filter sentences to only those with matching train data
    aligned_sentences = sentences_df[sentences_df['text_uuid'].isin(overlap)].copy()
    print(f"Sentences with train match: {len(aligned_sentences):,}")
    
    # Join with train data to get original document info
    aligned_df = aligned_sentences.merge(
        train_df[['oare_id', 'transliteration', 'translation']],
        left_on='text_uuid',
        right_on='oare_id',
        how='left',
        suffixes=('_sentence', '_document')
    )
    
    # Rename columns for clarity
    aligned_df = aligned_df.rename(columns={
        'translation': 'translation_sentence',
        'translation_document': 'document_translation',
        'transliteration': 'document_transliteration'
    })
    
    # Select and order relevant columns
    output_columns = [
        'text_uuid',
        'sentence_uuid',
        'display_name',
        'sentence_obj_in_text',
        'translation_sentence',
        'first_word_transcription',
        'first_word_spelling',
        'line_number',
        'side',
        'column',
        'document_transliteration',
        'document_translation'
    ]
    
    # Only keep columns that exist
    output_columns = [c for c in output_columns if c in aligned_df.columns]
    aligned_df = aligned_df[output_columns]
    
    print(f"Final aligned dataset: {len(aligned_df):,} sentence-level pairs")
    print(f"From {aligned_df['text_uuid'].nunique():,} unique documents")
    
    return aligned_df


def validate_alignment(aligned_df, sample_size=5):
    """Validate alignment quality on a sample."""
    print("\n" + "=" * 60)
    print("STEP 4: Validating Alignment Quality")
    print("=" * 60)
    
    # Get a sample of documents with multiple sentences
    doc_counts = aligned_df.groupby('text_uuid').size()
    multi_sentence_docs = doc_counts[doc_counts > 1].index.tolist()
    
    print(f"Documents with multiple sentences: {len(multi_sentence_docs):,}")
    
    # Show sample
    if multi_sentence_docs:
        sample_doc = multi_sentence_docs[0]
        sample = aligned_df[aligned_df['text_uuid'] == sample_doc].head(sample_size)
        
        print(f"\n--- Sample Document: {sample['display_name'].iloc[0]} ---")
        print(f"Document UUID: {sample_doc}")
        print(f"Number of sentences: {len(aligned_df[aligned_df['text_uuid'] == sample_doc])}")
        
        for idx, row in sample.iterrows():
            print(f"\n  Sentence {row['sentence_obj_in_text']}:")
            print(f"    Translation: {row['translation_sentence'][:100]}..." 
                  if len(str(row['translation_sentence'])) > 100 
                  else f"    Translation: {row['translation_sentence']}")
            if 'first_word_spelling' in row and pd.notna(row['first_word_spelling']):
                print(f"    First word: {row['first_word_spelling']}")
            if 'line_number' in row and pd.notna(row['line_number']):
                print(f"    Line: {row['line_number']}")
    
    # Quality metrics
    print("\n--- Quality Metrics ---")
    non_empty_translations = aligned_df['translation_sentence'].notna().sum()
    print(f"Non-empty sentence translations: {non_empty_translations:,} ({100*non_empty_translations/len(aligned_df):.1f}%)")
    
    avg_sentences_per_doc = len(aligned_df) / aligned_df['text_uuid'].nunique()
    print(f"Average sentences per document: {avg_sentences_per_doc:.2f}")
    
    return True


def save_dataset(aligned_df, output_path):
    """Save the aligned dataset."""
    print("\n" + "=" * 60)
    print("STEP 5: Saving Dataset")
    print("=" * 60)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    aligned_df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Rows: {len(aligned_df):,}")
    
    return output_path


def main():
    """Main execution flow."""
    print("\n" + "=" * 60)
    print("SENTENCE ALIGNMENT INTEGRATION")
    print("Phase 1, Task 1.1 - Deep Past Challenge")
    print("=" * 60 + "\n")
    
    # Load data
    sentences_df, train_df = load_data()
    
    # Analyze overlap
    overlap = analyze_overlap(sentences_df, train_df)
    
    # Create aligned dataset
    aligned_df = create_sentence_aligned_dataset(sentences_df, train_df, overlap)
    
    # Validate
    validate_alignment(aligned_df)
    
    # Save output
    output_path = OUTPUT_DIR / "sentence_aligned_train.csv"
    save_dataset(aligned_df, output_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input: {len(sentences_df):,} sentences, {len(train_df):,} training docs")
    print(f"Output: {len(aligned_df):,} aligned sentence pairs from {aligned_df['text_uuid'].nunique():,} docs")
    print(f"Expansion: {len(aligned_df)/len(train_df):.2f}x more training examples")
    print("=" * 60 + "\n")
    
    return aligned_df


if __name__ == "__main__":
    aligned_df = main()
