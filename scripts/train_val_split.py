#!/usr/bin/env python3
"""
Train/Validation Split for Deep Past Challenge

Creates a stratified train/validation split that:
1. Ensures ~10% validation set
2. Balances by document length (short/medium/long)
3. Balances by source type (document vs sentence)
4. Ensures no document ID overlap between splits

Usage:
    python scripts/train_val_split.py --input FILE --val-ratio 0.1
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# LENGTH BINNING
# =============================================================================

def create_length_bins(lengths: pd.Series, n_bins: int = 3) -> pd.Series:
    """
    Create length-based bins for stratification.
    
    Args:
        lengths: Series of text lengths
        n_bins: Number of bins (default 3: short/medium/long)
        
    Returns:
        Series with bin labels
    """
    # Use quantile-based bins
    bins = pd.qcut(lengths, q=n_bins, labels=['short', 'medium', 'long'], duplicates='drop')
    return bins


def get_length_quartile_boundaries(lengths: pd.Series) -> Dict[str, float]:
    """Get the quartile boundaries for reference."""
    return {
        'min': lengths.min(),
        'Q1': lengths.quantile(0.25),
        'Q2': lengths.quantile(0.50),
        'Q3': lengths.quantile(0.75),
        'max': lengths.max(),
    }


# =============================================================================
# STRATIFIED SPLIT
# =============================================================================

def stratified_split_by_document(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    source_col: str = 'source',
    id_col: str = 'oare_id',
    length_col: str = 'text_len',
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation split ensuring no document overlap.
    
    IMPORTANT: The same document ID may appear in multiple sources (e.g., a full
    document in 'train_document' and sentences from it in 'aligned_sentence').
    We must split by DOCUMENT ID first, then assign all rows with that ID to
    the same split.
    
    Stratification is done by:
    1. Computing document-level features (primary source, max length)
    2. Binning documents by length
    3. Splitting documents within each (source, length_bin) stratum
    4. Assigning ALL rows for each document to the appropriate split
    
    Args:
        df: Input DataFrame
        val_ratio: Fraction for validation (default 0.1)
        source_col: Column with source type
        id_col: Column with document ID
        length_col: Column with text length
        random_seed: Random seed for reproducibility
        
    Returns:
        (train_df, val_df) tuple
    """
    np.random.seed(random_seed)
    
    df = df.copy()
    
    # Step 1: Compute document-level features
    # For each document ID, get its primary source and max text length
    doc_features = df.groupby(id_col).agg({
        source_col: lambda x: x.mode().iloc[0],  # Most common source
        length_col: 'max'  # Maximum length (use document length if available)
    }).reset_index()
    doc_features.columns = [id_col, 'primary_source', 'max_length']
    
    # Step 2: Create length bins for documents
    doc_features['length_bin'] = create_length_bins(doc_features['max_length'])
    
    # Step 3: Create stratification key
    doc_features['strat_key'] = doc_features['primary_source'] + '_' + doc_features['length_bin'].astype(str)
    
    # Step 4: Split document IDs within each stratum
    train_ids = []
    val_ids = []
    
    for strat_key in doc_features['strat_key'].unique():
        stratum_docs = doc_features[doc_features['strat_key'] == strat_key][id_col].tolist()
        np.random.shuffle(stratum_docs)
        
        n_val = max(1, int(len(stratum_docs) * val_ratio))
        
        val_ids.extend(stratum_docs[:n_val])
        train_ids.extend(stratum_docs[n_val:])
    
    # Convert to sets for fast lookup
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    
    # Ensure no overlap (document can only be in one set)
    # If somehow a doc ended up in both (shouldn't happen), assign to train
    overlap = train_ids_set & val_ids_set
    if overlap:
        val_ids_set -= overlap
    
    # Step 5: Assign ALL rows for each document to the appropriate split
    train_df = df[df[id_col].isin(train_ids_set)].copy()
    val_df = df[df[id_col].isin(val_ids_set)].copy()
    
    return train_df, val_df


def verify_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, id_col: str) -> bool:
    """Verify there's no document ID overlap between train and validation."""
    train_ids = set(train_df[id_col].unique())
    val_ids = set(val_df[id_col].unique())
    overlap = train_ids & val_ids
    
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping document IDs!")
        return False
    return True


# =============================================================================
# STATISTICS AND REPORTING
# =============================================================================

def compute_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    source_col: str = 'source',
    id_col: str = 'oare_id',
    length_col: str = 'text_len'
) -> Dict:
    """Compute statistics about the train/val split."""
    
    stats = {
        'train': {
            'total_rows': len(train_df),
            'unique_docs': train_df[id_col].nunique(),
            'source_distribution': train_df[source_col].value_counts().to_dict(),
            'length_stats': {
                'mean': float(train_df[length_col].mean()),
                'std': float(train_df[length_col].std()),
                'min': int(train_df[length_col].min()),
                'max': int(train_df[length_col].max()),
                'median': float(train_df[length_col].median()),
            }
        },
        'validation': {
            'total_rows': len(val_df),
            'unique_docs': val_df[id_col].nunique(),
            'source_distribution': val_df[source_col].value_counts().to_dict(),
            'length_stats': {
                'mean': float(val_df[length_col].mean()),
                'std': float(val_df[length_col].std()),
                'min': int(val_df[length_col].min()),
                'max': int(val_df[length_col].max()),
                'median': float(val_df[length_col].median()),
            }
        },
        'split_ratio': {
            'train_pct': len(train_df) / (len(train_df) + len(val_df)) * 100,
            'val_pct': len(val_df) / (len(train_df) + len(val_df)) * 100,
        }
    }
    
    # Add per-source split ratios
    all_sources = set(train_df[source_col].unique()) | set(val_df[source_col].unique())
    stats['per_source_ratio'] = {}
    for source in all_sources:
        train_count = len(train_df[train_df[source_col] == source])
        val_count = len(val_df[val_df[source_col] == source])
        total = train_count + val_count
        if total > 0:
            stats['per_source_ratio'][source] = {
                'train': train_count,
                'val': val_count,
                'val_pct': val_count / total * 100
            }
    
    return stats


def print_split_report(stats: Dict):
    """Print a formatted report of the split statistics."""
    
    print("\n" + "=" * 60)
    print("TRAIN/VALIDATION SPLIT REPORT")
    print("=" * 60)
    
    print(f"\n📊 Overall Split:")
    print(f"  Train: {stats['train']['total_rows']:,} rows ({stats['split_ratio']['train_pct']:.1f}%)")
    print(f"  Val:   {stats['validation']['total_rows']:,} rows ({stats['split_ratio']['val_pct']:.1f}%)")
    
    print(f"\n📄 Unique Documents:")
    print(f"  Train: {stats['train']['unique_docs']:,} documents")
    print(f"  Val:   {stats['validation']['unique_docs']:,} documents")
    
    print(f"\n📈 Source Distribution:")
    print(f"  {'Source':<20} {'Train':>8} {'Val':>8} {'Val %':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for source, data in stats['per_source_ratio'].items():
        print(f"  {source:<20} {data['train']:>8,} {data['val']:>8,} {data['val_pct']:>7.1f}%")
    
    print(f"\n📏 Length Statistics:")
    print(f"  {'Metric':<15} {'Train':>12} {'Val':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for metric in ['mean', 'median', 'min', 'max']:
        train_val = stats['train']['length_stats'][metric]
        val_val = stats['validation']['length_stats'][metric]
        print(f"  {metric:<15} {train_val:>12.1f} {val_val:>12.1f}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create stratified train/validation split for Deep Past Challenge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', type=str,
                       default='data/processed/final_training_data_cleaned_normalized.csv',
                       help='Input CSV file')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='data/processed',
                       help='Output directory for split files')
    parser.add_argument('--val-ratio', '-v', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--text-col', type=str, default='transliteration_cleaned',
                       help='Column for text length calculation')
    parser.add_argument('--source-col', type=str, default='source',
                       help='Column with source type')
    parser.add_argument('--id-col', type=str, default='oare_id',
                       help='Column with document ID')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_dir = project_root / args.output_dir
    
    print("=" * 60)
    print("TRAIN/VALIDATION SPLIT")
    print("Phase 3.4 - Deep Past Challenge")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading: {input_path}")
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df):,} rows")
    
    # Calculate text length if not present
    if 'text_len' not in df.columns:
        df['text_len'] = df[args.text_col].str.len()
    
    # Print input statistics
    print(f"\n📊 Input Data Statistics:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique documents: {df[args.id_col].nunique():,}")
    print(f"  Sources: {dict(df[args.source_col].value_counts())}")
    
    # Get length quartiles
    quartiles = get_length_quartile_boundaries(df['text_len'])
    print(f"\n📏 Text Length Distribution:")
    for name, val in quartiles.items():
        print(f"  {name}: {val:.0f}")
    
    # Create stratified split
    print(f"\n🔄 Creating stratified split (val_ratio={args.val_ratio}, seed={args.seed})...")
    
    train_df, val_df = stratified_split_by_document(
        df,
        val_ratio=args.val_ratio,
        source_col=args.source_col,
        id_col=args.id_col,
        length_col='text_len',
        random_seed=args.seed
    )
    
    # Verify no overlap
    print("\n✓ Verifying no document overlap...")
    if verify_no_overlap(train_df, val_df, args.id_col):
        print("  ✓ No overlap found between train and validation sets")
    else:
        print("  ✗ ERROR: Overlap detected!")
        return None
    
    # Compute and print statistics
    stats = compute_split_statistics(
        train_df, val_df,
        source_col=args.source_col,
        id_col=args.id_col,
        length_col='text_len'
    )
    print_split_report(stats)
    
    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Drop text_len column before saving
    if 'text_len' in train_df.columns:
        train_df = train_df.drop(columns=['text_len'])
        val_df = val_df.drop(columns=['text_len'])
    
    # Save train set
    train_path = output_dir / 'train_final.csv'
    train_df.to_csv(train_path, index=False)
    print(f"✓ Saved train set: {train_path}")
    print(f"  {len(train_df):,} rows, {train_df[args.id_col].nunique():,} documents")
    
    # Save validation set
    val_path = output_dir / 'val_final.csv'
    val_df.to_csv(val_path, index=False)
    print(f"✓ Saved validation set: {val_path}")
    print(f"  {len(val_df):,} rows, {val_df[args.id_col].nunique():,} documents")
    
    # Save split statistics
    stats_path = output_dir / 'train_val_split_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved statistics: {stats_path}")
    
    # Save document ID lists for reference
    ids_path = output_dir / 'train_val_document_ids.json'
    id_lists = {
        'train_ids': sorted(train_df[args.id_col].unique().tolist()),
        'val_ids': sorted(val_df[args.id_col].unique().tolist()),
    }
    with open(ids_path, 'w', encoding='utf-8') as f:
        json.dump(id_lists, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved document IDs: {ids_path}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60 + "\n")
    
    return stats


if __name__ == '__main__':
    main()
