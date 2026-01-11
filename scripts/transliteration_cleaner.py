#!/usr/bin/env python3
"""
Transliteration Cleaning Pipeline for Deep Past Challenge

This script normalizes Akkadian transliterations by:
1. Removing scribal annotations (!, ?, ˹ ˺)
2. Standardizing gaps ([x] → <gap>, … → <big_gap>)
3. Normalizing brackets ([ ] → content only)
4. Handling determinatives ({d}, {ki}, etc.) consistently

Based on the formatting suggestions from the competition overview.

Usage:
    python scripts/transliteration_cleaner.py [--input FILE] [--output FILE] [--verbose]
    
Example:
    python scripts/transliteration_cleaner.py --input data/train.csv --output data/processed/train_cleaned.csv
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

# Valid Akkadian determinatives (from competition documentation)
VALID_DETERMINATIVES = {
    'd',      # dingir 'god, deity'
    'mul',    # 'stars'
    'ki',     # 'earth' (location)
    'lu₂',    # LÚ (people/professions)
    'lu2',    # alternate form
    'e₂',     # É (buildings)
    'e2',     # alternate form
    'uru',    # URU (settlements)
    'kur',    # KUR (lands/mountains)
    'mi',     # munus (f) (feminine names)
    'm',      # (1 or m) (masculine names)
    'f',      # feminine marker
    'geš',    # GIŠ (wood/trees)
    'ĝeš',    # alternate form
    'tug₂',   # TÚG (textiles)
    'tug2',   # alternate form
    'dub',    # DUB (tablets/documents)
    'id₂',    # ÍD (rivers/canals)
    'id2',    # alternate form
    'mušen',  # MUŠEN (birds)
    'na₄',    # na4 (stone)
    'na4',    # alternate form
    'kuš',    # kuš (hides/skin)
    'u₂',     # Ú (plants)
    'u2',     # alternate form
    'giš',    # alternate for geš
    'urudu',  # copper
    'zabar',  # bronze
}


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def remove_scribal_annotations(text: str) -> str:
    """
    Remove modern scribal annotations from transliteration.
    
    Removes:
    - ! (certain reading of a difficult sign)
    - ? (uncertain reading)
    - ˹ ˺ (partially broken signs - half brackets)
    - / (signs found below the line / line divider)
    - : (word divider)
    - << >> (errant/erroneous signs and their content)
    
    Preserves but cleans:
    - < > (scribal insertions - keep text inside)
    """
    if not text or pd.isna(text):
        return ""
    
    # Remove errant/erroneous signs completely (with their content)
    text = re.sub(r'<<[^>]*>>', '', text)
    
    # Remove scribal insertion markers but keep content: <text> → text
    text = re.sub(r'<([^<>]+)>', r'\1', text)
    
    # Remove half brackets (partially broken signs): ˹text˺ → text
    text = re.sub(r'˹([^˺]*)˺', r'\1', text)
    # Also handle standalone half brackets
    text = text.replace('˹', '').replace('˺', '')
    
    # Remove certain reading marker
    text = text.replace('!', '')
    
    # Remove uncertain reading marker
    text = re.sub(r'\?(?![^\[]*\])', '', text)  # Don't remove ? inside brackets like [x?]
    
    # Remove line divider
    text = text.replace('/', ' ')
    
    # Remove word divider (colon used as divider)
    # Be careful not to remove colons that are part of logograms like KÙ.BABBAR
    text = re.sub(r'(?<![A-ZŠṢṬḪa-zšṣṭḫ]): *', ' ', text)
    
    return text


def standardize_gaps(text: str) -> str:
    """
    Standardize gap markers in transliteration.
    
    Converts:
    - [x] → <gap> (single unknown sign)
    - [x x] or [x x x] → <big_gap> (multiple unknown signs)
    - … or ... → <big_gap> (ellipsis indicating large break)
    - [… …] or [... ...] → <big_gap>
    - (broken) or (damaged) markers → <gap> or <big_gap>
    """
    if not text or pd.isna(text):
        return ""
    
    # Handle [... ...] or [… …] patterns first (large gaps)
    text = re.sub(r'\[\s*[\.…]+\s*[\.…]*\s*\]', '<big_gap>', text)
    
    # Handle multiple x patterns: [x x], [x x x], etc. → <big_gap>
    text = re.sub(r'\[\s*x(\s+x)+\s*\]', '<big_gap>', text, flags=re.IGNORECASE)
    
    # Handle single [x] → <gap>
    text = re.sub(r'\[\s*x\s*\]', '<gap>', text, flags=re.IGNORECASE)
    
    # Handle ... ellipsis → <big_gap>
    text = re.sub(r'\.{3,}', '<big_gap>', text)
    
    # Handle … (unicode ellipsis) → <big_gap>
    text = text.replace('…', '<big_gap>')
    
    # Handle (broken), (damaged), (rest broken), etc.
    text = re.sub(r'\(\s*(?:rest\s+)?(?:broken|damaged|destroyed|missing|illegible)\s*\)', 
                  '<big_gap>', text, flags=re.IGNORECASE)
    
    # Handle (x lines broken/missing)
    text = re.sub(r'\(\s*\d+\s*(?:lines?\s+)?(?:broken|missing|destroyed)\s*\)', 
                  '<big_gap>', text, flags=re.IGNORECASE)
    
    # Clean up multiple consecutive gaps (but preserve single gaps)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    text = re.sub(r'(<big_gap>\s*){2,}', '<big_gap> ', text)
    text = re.sub(r'<gap>\s*<big_gap>', '<big_gap>', text)
    text = re.sub(r'<big_gap>\s*<gap>', '<big_gap>', text)
    
    return text


def normalize_brackets(text: str) -> str:
    """
    Normalize square brackets in transliteration.
    
    For document-level transliteration:
    - [text] → text (remove brackets, keep content)
    - Brackets indicate restored/damaged text but content is valid
    
    Note: This should be called AFTER standardize_gaps() to preserve gap markers.
    """
    if not text or pd.isna(text):
        return ""
    
    # Remove brackets but keep content
    # First protect our gap markers
    text = text.replace('<gap>', '<<<GAP>>>')
    text = text.replace('<big_gap>', '<<<BIG_GAP>>>')
    
    # Remove brackets around content: [KÙ.BABBAR] → KÙ.BABBAR
    text = re.sub(r'\[([^\[\]]+)\]', r'\1', text)
    
    # Handle nested brackets (rare but possible)
    while '[' in text and ']' in text:
        text = re.sub(r'\[([^\[\]]+)\]', r'\1', text)
    
    # Restore gap markers
    text = text.replace('<<<GAP>>>', '<gap>')
    text = text.replace('<<<BIG_GAP>>>', '<big_gap>')
    
    return text


def normalize_determinatives(text: str) -> str:
    """
    Ensure determinatives are consistently formatted in curly brackets.
    
    - Validates existing {det} markers
    - Normalizes variant forms (e.g., {lu2} → {lu₂})
    - Handles superscript determinatives that may appear differently
    """
    if not text or pd.isna(text):
        return ""
    
    # Normalize subscript numbers in determinatives
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    }
    
    def normalize_det_content(match):
        """Normalize the content inside determinative brackets."""
        content = match.group(1).lower()
        
        # Convert regular numbers to subscript at end of determinative
        content = re.sub(r'(\d+)$', 
                        lambda m: ''.join(subscript_map.get(c, c) for c in m.group(1)), 
                        content)
        
        return '{' + content + '}'
    
    # Normalize all determinatives
    text = re.sub(r'\{([^}]+)\}', normalize_det_content, text)
    
    return text


def remove_line_numbers(text: str) -> str:
    """
    Remove line numbers from transliteration.
    
    Line numbers typically appear as: 1. text, 5. text, 1'. text, etc.
    """
    if not text or pd.isna(text):
        return ""
    
    # Remove line numbers at start of lines: "1. text" → "text"
    # Handle primes: 1', 1'', etc.
    text = re.sub(r"^\s*\d+['\u2033\u2032\u2019]*\.\s*", '', text, flags=re.MULTILINE)
    
    # Remove line numbers in the middle that look like: " 5. "
    text = re.sub(r"\s+\d+['\u2033\u2032\u2019]*\.\s+", ' ', text)
    
    return text


def clean_whitespace(text: str) -> str:
    """
    Clean up whitespace in the text.
    
    - Normalize multiple spaces to single space
    - Strip leading/trailing whitespace
    - Remove spaces around hyphens (for Akkadian syllables)
    """
    if not text or pd.isna(text):
        return ""
    
    # Normalize spaces around hyphens: "a - na" → "a-na"
    text = re.sub(r'\s*-\s*', '-', text)
    
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    return text


def remove_parenthetical_comments(text: str) -> str:
    """
    Remove or handle parenthetical comments.
    
    - (erasure) → remove entirely
    - (x signs) → <gap> or <big_gap>
    - Keep meaningful content if present
    """
    if not text or pd.isna(text):
        return ""
    
    # Remove erasure markers
    text = re.sub(r'\(\s*erasure\s*\)', '', text, flags=re.IGNORECASE)
    
    # Remove blank/uninscribed markers
    text = re.sub(r'\(\s*(?:blank|uninscribed|space)\s*\)', '', text, flags=re.IGNORECASE)
    
    # Handle (x signs), (n signs broken), etc.
    text = re.sub(r'\(\s*\d+\s*signs?\s*(?:broken|missing|destroyed)?\s*\)', 
                  '<gap>', text, flags=re.IGNORECASE)
    
    return text


# =============================================================================
# MAIN CLEANING PIPELINE
# =============================================================================

def clean_transliteration(text: str, verbose: bool = False) -> str:
    """
    Apply full cleaning pipeline to a transliteration.
    
    Order matters! Steps are applied in sequence:
    1. Remove line numbers
    2. Remove scribal annotations
    3. Remove parenthetical comments
    4. Standardize gaps (before bracket normalization)
    5. Normalize brackets
    6. Normalize determinatives
    7. Clean whitespace
    
    Args:
        text: Raw transliteration text
        verbose: If True, print intermediate steps
        
    Returns:
        Cleaned transliteration text
    """
    if not text or pd.isna(text):
        return ""
    
    original = text
    
    # Step 1: Remove line numbers
    text = remove_line_numbers(text)
    if verbose and text != original:
        print(f"  After line numbers: {text[:100]}...")
    
    # Step 2: Remove scribal annotations
    text = remove_scribal_annotations(text)
    if verbose:
        print(f"  After scribal annotations: {text[:100]}...")
    
    # Step 3: Remove parenthetical comments
    text = remove_parenthetical_comments(text)
    if verbose:
        print(f"  After parenthetical: {text[:100]}...")
    
    # Step 4: Standardize gaps (MUST come before bracket normalization)
    text = standardize_gaps(text)
    if verbose:
        print(f"  After gaps: {text[:100]}...")
    
    # Step 5: Normalize brackets
    text = normalize_brackets(text)
    if verbose:
        print(f"  After brackets: {text[:100]}...")
    
    # Step 6: Normalize determinatives
    text = normalize_determinatives(text)
    if verbose:
        print(f"  After determinatives: {text[:100]}...")
    
    # Step 7: Clean whitespace
    text = clean_whitespace(text)
    
    return text


def clean_dataframe(df: pd.DataFrame, 
                   transliteration_col: str = 'transliteration',
                   output_col: Optional[str] = None,
                   verbose: bool = False) -> pd.DataFrame:
    """
    Apply cleaning pipeline to a DataFrame.
    
    Args:
        df: DataFrame with transliteration column
        transliteration_col: Name of column containing transliterations
        output_col: Name for cleaned column (default: adds '_cleaned' suffix)
        verbose: If True, print progress
        
    Returns:
        DataFrame with cleaned transliteration column added
    """
    if output_col is None:
        output_col = f"{transliteration_col}_cleaned"
    
    df = df.copy()
    
    if verbose:
        print(f"Cleaning {len(df):,} transliterations...")
    
    # Apply cleaning
    df[output_col] = df[transliteration_col].apply(
        lambda x: clean_transliteration(x, verbose=False)
    )
    
    if verbose:
        # Show sample of changes
        changed = df[df[transliteration_col] != df[output_col]]
        print(f"  Modified: {len(changed):,} rows ({100*len(changed)/len(df):.1f}%)")
        
        # Show a few examples
        if len(changed) > 0:
            print("\n  Sample changes:")
            for idx in changed.head(3).index:
                orig = df.loc[idx, transliteration_col][:80]
                clean = df.loc[idx, output_col][:80]
                print(f"    Original: {orig}...")
                print(f"    Cleaned:  {clean}...")
                print()
    
    return df


def compute_cleaning_stats(df: pd.DataFrame,
                          original_col: str = 'transliteration',
                          cleaned_col: str = 'transliteration_cleaned') -> dict:
    """
    Compute statistics about the cleaning process.
    """
    stats = {
        'total_rows': len(df),
        'rows_modified': (df[original_col] != df[cleaned_col]).sum(),
        'avg_original_len': df[original_col].str.len().mean(),
        'avg_cleaned_len': df[cleaned_col].str.len().mean(),
        'gap_count': df[cleaned_col].str.count('<gap>').sum(),
        'big_gap_count': df[cleaned_col].str.count('<big_gap>').sum(),
    }
    
    # Count determinatives
    det_pattern = r'\{[^}]+\}'
    stats['determinative_count'] = df[cleaned_col].str.count(det_pattern).sum()
    
    # Unique determinatives
    all_dets = []
    for text in df[cleaned_col].dropna():
        all_dets.extend(re.findall(det_pattern, text))
    stats['unique_determinatives'] = len(set(all_dets))
    
    return stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Clean Akkadian transliterations for Deep Past Challenge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', type=str, default='data/train.csv',
                       help='Input CSV file path (default: data/train.csv)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output CSV file path (default: data/processed/<input>_cleaned.csv)')
    parser.add_argument('--column', '-c', type=str, default='transliteration',
                       help='Column name containing transliterations (default: transliteration)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress')
    parser.add_argument('--sample', type=int, default=None,
                       help='Process only N rows (for testing)')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    
    if args.output:
        output_path = project_root / args.output
    else:
        output_dir = project_root / 'data' / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        output_path = output_dir / f'{stem}_cleaned.csv'
    
    print("=" * 60)
    print("TRANSLITERATION CLEANING PIPELINE")
    print("Phase 2, Task 2.1 - Deep Past Challenge")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading: {input_path}")
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        print("Please ensure the data file exists.")
        return None
    
    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df):,} rows")
    
    if args.sample:
        df = df.head(args.sample)
        print(f"  Using sample: {len(df):,} rows")
    
    # Check column exists
    if args.column not in df.columns:
        print(f"ERROR: Column '{args.column}' not found in {input_path}")
        print(f"  Available columns: {list(df.columns)}")
        return None
    
    # Clean
    print(f"\nCleaning column: {args.column}")
    df = clean_dataframe(df, transliteration_col=args.column, verbose=args.verbose)
    
    # Statistics
    print("\n" + "=" * 60)
    print("CLEANING STATISTICS")
    print("=" * 60)
    
    stats = compute_cleaning_stats(df, original_col=args.column)
    print(f"  Total rows:          {stats['total_rows']:,}")
    print(f"  Rows modified:       {stats['rows_modified']:,} ({100*stats['rows_modified']/stats['total_rows']:.1f}%)")
    print(f"  Avg original length: {stats['avg_original_len']:.1f} chars")
    print(f"  Avg cleaned length:  {stats['avg_cleaned_len']:.1f} chars")
    print(f"  Total <gap> markers: {stats['gap_count']:,}")
    print(f"  Total <big_gap> markers: {stats['big_gap_count']:,}")
    print(f"  Total determinatives: {stats['determinative_count']:,}")
    print(f"  Unique determinatives: {stats['unique_determinatives']}")
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Show sample of cleaned text
    if args.verbose:
        print("\n" + "=" * 60)
        print("SAMPLE CLEANED TRANSLITERATIONS")
        print("=" * 60)
        for i, row in df.head(3).iterrows():
            print(f"\n--- Row {i} ---")
            print(f"Original: {row[args.column][:150]}...")
            print(f"Cleaned:  {row[f'{args.column}_cleaned'][:150]}...")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60 + "\n")
    
    return df


if __name__ == '__main__':
    main()
