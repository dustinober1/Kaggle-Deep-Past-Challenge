#!/usr/bin/env python3
"""
Translation Normalizer for Deep Past Challenge

This script normalizes English translations in the dataset:
1. Standardize ellipsis markers (... → …)
2. Normalize quotation marks (consistent double quotes)
3. Consistent handling of [broken] text markers
4. Standardize proper noun capitalization

Usage:
    python scripts/translation_normalizer.py --input FILE --output FILE [--options]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =============================================================================
# ELLIPSIS NORMALIZATION
# =============================================================================

def standardize_ellipsis(text: str) -> str:
    """
    Standardize all forms of ellipsis to a single Unicode ellipsis (…).
    
    Converts:
    - ... (three dots) → …
    - .. (two dots) → …
    - .... (four+ dots) → …
    - Multiple spaces around ellipsis → single space
    """
    if not text or pd.isna(text):
        return ""
    
    # Replace 2+ consecutive dots with ellipsis
    text = re.sub(r'\.{2,}', '…', text)
    
    # Clean up multiple ellipses in a row
    text = re.sub(r'…+', '…', text)
    
    # Clean up spaces around ellipsis (but keep at least one)
    text = re.sub(r'\s*…\s*', ' … ', text)
    
    # Clean up ellipsis at start/end of string
    text = re.sub(r'^\s*…\s*', '… ', text)
    text = re.sub(r'\s*…\s*$', ' …', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()


# =============================================================================
# QUOTATION MARK NORMALIZATION
# =============================================================================

def normalize_quotation_marks(text: str) -> str:
    """
    Normalize all quotation marks to standard ASCII double quotes.
    
    Converts:
    - " " (smart quotes) → "
    - '' (two single quotes) → "
    - « » (guillemets) → "
    - `` '' (TeX-style) → "
    
    Also cleans up:
    - Extra quotes at end of sentences ("") → "
    - Mismatched quotes
    """
    if not text or pd.isna(text):
        return ""
    
    # Convert smart quotes to standard
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    
    # Convert guillemets
    text = text.replace('«', '"')
    text = text.replace('»', '"')
    
    # Convert TeX-style quotes
    text = text.replace('``', '"')
    text = text.replace("''", '"')
    
    # Clean up doubled quotes at end of sentences
    # Pattern: word.""  → word."
    text = re.sub(r'([.!?])""+', r'\1"', text)
    
    # Clean up tripled+ quotes
    text = re.sub(r'"{3,}', '"', text)
    
    # Clean up doubled quotes (but preserve intentional "" for empty)
    text = re.sub(r'"\s*"(?=[A-Za-z])', '" "', text)
    
    return text


# =============================================================================
# BRACKET TEXT HANDLING
# =============================================================================

def standardize_brackets(text: str) -> str:
    """
    Standardize handling of text in square brackets.
    
    Conventions:
    - [x] or [xxx] → [?] (uncertain reading)
    - [broken] → [damaged]
    - [...] → [gap]
    - [gave] (restored text) → keep as-is
    - [and Assu]r (partial restoration) → keep as-is
    
    Note: We keep editorial markers but standardize terminology.
    """
    if not text or pd.isna(text):
        return ""
    
    # Standardize gap markers
    text = re.sub(r'\[\s*\.\.\.\s*\]', '[gap]', text)
    text = re.sub(r'\[\s*…\s*\]', '[gap]', text)
    
    # Standardize uncertain readings: [x], [xx], [xxx], [x x x]
    text = re.sub(r'\[\s*x+(?:\s+x+)*\s*\]', '[?]', text, flags=re.IGNORECASE)
    
    # Standardize broken/damaged markers
    text = re.sub(r'\[\s*broken\s*\]', '[damaged]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*lost\s*\]', '[damaged]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*illegible\s*\]', '[damaged]', text, flags=re.IGNORECASE)
    
    return text


def extract_bracket_patterns(text: str) -> List[str]:
    """Extract all bracket patterns for analysis."""
    if not text or pd.isna(text):
        return []
    
    return re.findall(r'\[[^\]]+\]', text)


# =============================================================================
# PROPER NOUN CAPITALIZATION
# =============================================================================

# Known proper nouns in Old Assyrian translations
KNOWN_PROPER_NOUNS = {
    # Deities
    'aššur', 'ištar', 'šamaš', 'adad', 'sin', 'enlil', 'illil',
    # Places  
    'kanesh', 'kaneš', 'assur', 'aššur', 'burušhaddum', 'durhumit',
    'wahšušana', 'purušhaddum', 'nahur', 'zalpa', 'amkuwa',
    # Titles/Terms that should be capitalized
    'city', 'palace', 'colony',
}


def standardize_proper_nouns(text: str) -> str:
    """
    Ensure proper nouns are consistently capitalized.
    
    Note: This is conservative - only fixes clearly lowercase proper nouns.
    Akkadian proper names already vary in transcription conventions.
    """
    if not text or pd.isna(text):
        return ""
    
    # Capitalize "City" when it refers to Aššur city
    text = re.sub(r'\bthe city\b(?!\s+of)', 'the City', text)
    text = re.sub(r'\bCity Hall\b', 'City Hall', text, flags=re.IGNORECASE)
    
    # Ensure Aššur is capitalized (the deity/city)
    text = re.sub(r"\baššur's\b", "Aššur's", text)
    text = re.sub(r'\baššur\b', 'Aššur', text)
    
    # Capitalize Kanesh
    text = re.sub(r'\bkanesh\b', 'Kanesh', text, flags=re.IGNORECASE)
    text = re.sub(r'\bkaneš\b', 'Kaneš', text, flags=re.IGNORECASE)
    
    return text


# =============================================================================
# WHITESPACE AND PUNCTUATION CLEANUP
# =============================================================================

def clean_whitespace(text: str) -> str:
    """Clean up whitespace issues."""
    if not text or pd.isna(text):
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Fix space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Fix missing space after punctuation (except decimals)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    
    # Don't add space after decimal points in numbers
    text = re.sub(r'(\d)\. (\d)', r'\1.\2', text)
    
    return text


def clean_punctuation(text: str) -> str:
    """Clean up punctuation issues."""
    if not text or pd.isna(text):
        return ""
    
    # Remove double periods (but not ellipsis)
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text)
    
    # Fix spacing around dashes
    text = re.sub(r'\s*-\s*', '-', text)
    text = re.sub(r'(\w)-(\w)', r'\1-\2', text)
    
    # Fix spacing around slashes
    text = re.sub(r'\s*/\s*', ' / ', text)
    
    return text


# =============================================================================
# MAIN NORMALIZATION PIPELINE
# =============================================================================

def normalize_translation(text: str, 
                         standardize_ellipses: bool = True,
                         normalize_quotes: bool = True,
                         standardize_bracket_text: bool = True,
                         fix_proper_nouns: bool = True,
                         clean_ws: bool = True) -> str:
    """
    Apply all normalization steps to a translation.
    
    Args:
        text: Input translation text
        standardize_ellipses: Convert ... to …
        normalize_quotes: Normalize quotation marks
        standardize_bracket_text: Standardize [broken] etc.
        fix_proper_nouns: Fix capitalization of known proper nouns
        clean_ws: Clean whitespace and punctuation
        
    Returns:
        Normalized translation text
    """
    if not text or pd.isna(text):
        return ""
    
    # Step 1: Normalize quotes first (affects other parsing)
    if normalize_quotes:
        text = normalize_quotation_marks(text)
    
    # Step 2: Standardize ellipsis
    if standardize_ellipses:
        text = standardize_ellipsis(text)
    
    # Step 3: Standardize bracket markers
    if standardize_bracket_text:
        text = standardize_brackets(text)
    
    # Step 4: Fix proper noun capitalization
    if fix_proper_nouns:
        text = standardize_proper_nouns(text)
    
    # Step 5: Clean whitespace and punctuation
    if clean_ws:
        text = clean_whitespace(text)
        text = clean_punctuation(text)
    
    return text


def analyze_translations(texts: List[str]) -> Dict:
    """
    Analyze translation patterns for reporting.
    
    Returns statistics about ellipsis usage, quotes, brackets, etc.
    """
    stats = {
        'total_texts': len(texts),
        'texts_with_ellipsis_dots': 0,
        'texts_with_ellipsis_unicode': 0,
        'texts_with_smart_quotes': 0,
        'texts_with_brackets': 0,
        'bracket_patterns': Counter(),
        'quote_patterns': Counter(),
    }
    
    for text in texts:
        if not text or pd.isna(text):
            continue
            
        # Check ellipsis patterns
        if '...' in text:
            stats['texts_with_ellipsis_dots'] += 1
        if '…' in text:
            stats['texts_with_ellipsis_unicode'] += 1
            
        # Check smart quotes
        if '"' in text or '"' in text or ''' in text or ''' in text:
            stats['texts_with_smart_quotes'] += 1
            
        # Check brackets
        brackets = extract_bracket_patterns(text)
        if brackets:
            stats['texts_with_brackets'] += 1
            stats['bracket_patterns'].update(brackets)
            
        # Check quote patterns
        quotes = re.findall(r'[""][^""]*[""]', text)
        stats['quote_patterns'].update(quotes[:5])  # Limit per text
    
    # Convert counters to dicts for JSON
    stats['bracket_patterns'] = dict(stats['bracket_patterns'].most_common(30))
    stats['quote_patterns'] = dict(stats['quote_patterns'].most_common(20))
    
    return stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Translation Normalizer for Deep Past Challenge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', type=str,
                       default='data/processed/train_cleaned.csv',
                       help='Input CSV file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output CSV file (default: adds _normalized suffix)')
    parser.add_argument('--column', '-c', type=str, default='translation',
                       help='Column containing translations')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze, do not modify data')
    parser.add_argument('--no-ellipsis', action='store_true',
                       help='Skip ellipsis standardization')
    parser.add_argument('--no-quotes', action='store_true',
                       help='Skip quotation mark normalization')
    parser.add_argument('--no-brackets', action='store_true',
                       help='Skip bracket standardization')
    parser.add_argument('--no-proper-nouns', action='store_true',
                       help='Skip proper noun capitalization')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    
    if args.output:
        output_path = project_root / args.output
    else:
        # Default: add _normalized suffix
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_normalized.csv"
    
    print("=" * 60)
    print("TRANSLATION NORMALIZER")
    print("Phase 3.3 - Deep Past Challenge")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading: {input_path}")
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df):,} rows")
    
    # Check column exists
    if args.column not in df.columns:
        print(f"ERROR: Column '{args.column}' not found")
        print(f"  Available columns: {list(df.columns)}")
        return None
    
    # Analyze before normalization
    print("\n" + "=" * 60)
    print("BEFORE NORMALIZATION")
    print("=" * 60)
    
    texts = df[args.column].dropna().tolist()
    before_stats = analyze_translations(texts)
    
    print(f"\n📊 Translation Statistics:")
    print(f"  Total translations: {before_stats['total_texts']:,}")
    print(f"  With '...' ellipsis: {before_stats['texts_with_ellipsis_dots']:,}")
    print(f"  With '…' ellipsis: {before_stats['texts_with_ellipsis_unicode']:,}")
    print(f"  With smart quotes: {before_stats['texts_with_smart_quotes']:,}")
    print(f"  With brackets: {before_stats['texts_with_brackets']:,}")
    
    print(f"\n📝 Top Bracket Patterns:")
    for i, (pattern, count) in enumerate(list(before_stats['bracket_patterns'].items())[:15], 1):
        print(f"  {i:2}. {pattern:25} ({count:,})")
    
    if args.analyze_only:
        print("\n(Analyze-only mode, not modifying data)")
        return before_stats
    
    # Apply normalization
    print("\n" + "=" * 60)
    print("NORMALIZING TRANSLATIONS")
    print("=" * 60)
    
    output_col = 'translation_normalized'
    
    # Count changes
    changes = {
        'ellipsis': 0,
        'quotes': 0,
        'brackets': 0,
        'proper_nouns': 0,
        'whitespace': 0,
    }
    
    def normalize_with_tracking(text):
        if not text or pd.isna(text):
            return ""
        
        original = text
        result = text
        
        # Track each change type
        if not args.no_quotes:
            after_quotes = normalize_quotation_marks(result)
            if after_quotes != result:
                changes['quotes'] += 1
            result = after_quotes
        
        if not args.no_ellipsis:
            after_ellipsis = standardize_ellipsis(result)
            if after_ellipsis != result:
                changes['ellipsis'] += 1
            result = after_ellipsis
        
        if not args.no_brackets:
            after_brackets = standardize_brackets(result)
            if after_brackets != result:
                changes['brackets'] += 1
            result = after_brackets
        
        if not args.no_proper_nouns:
            after_nouns = standardize_proper_nouns(result)
            if after_nouns != result:
                changes['proper_nouns'] += 1
            result = after_nouns
        
        after_ws = clean_whitespace(result)
        after_ws = clean_punctuation(after_ws)
        if after_ws != result:
            changes['whitespace'] += 1
        result = after_ws
        
        return result
    
    df[output_col] = df[args.column].apply(normalize_with_tracking)
    
    # Report changes
    total_changed = sum(1 for i, row in df.iterrows() 
                       if row[args.column] != row[output_col])
    
    print(f"\n✓ Normalization complete:")
    print(f"  Total rows modified: {total_changed:,} / {len(df):,}")
    print(f"  Ellipsis changes: {changes['ellipsis']:,}")
    print(f"  Quote changes: {changes['quotes']:,}")
    print(f"  Bracket changes: {changes['brackets']:,}")
    print(f"  Proper noun changes: {changes['proper_nouns']:,}")
    print(f"  Whitespace changes: {changes['whitespace']:,}")
    
    # Analyze after normalization
    print("\n" + "=" * 60)
    print("AFTER NORMALIZATION")
    print("=" * 60)
    
    after_texts = df[output_col].dropna().tolist()
    after_stats = analyze_translations(after_texts)
    
    print(f"\n📊 After Statistics:")
    print(f"  With '...' ellipsis: {after_stats['texts_with_ellipsis_dots']:,} (was {before_stats['texts_with_ellipsis_dots']:,})")
    print(f"  With '…' ellipsis: {after_stats['texts_with_ellipsis_unicode']:,} (was {before_stats['texts_with_ellipsis_unicode']:,})")
    print(f"  With smart quotes: {after_stats['texts_with_smart_quotes']:,} (was {before_stats['texts_with_smart_quotes']:,})")
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    # Save statistics
    stats_path = output_path.parent / 'translation_normalization_stats.json'
    all_stats = {
        'before': before_stats,
        'after': after_stats,
        'changes': changes,
        'total_modified': total_changed,
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved statistics: {stats_path}")
    
    # Show examples
    if args.verbose:
        print("\n" + "=" * 60)
        print("EXAMPLE CHANGES")
        print("=" * 60)
        
        changed_rows = df[df[args.column] != df[output_col]].head(5)
        for i, (_, row) in enumerate(changed_rows.iterrows(), 1):
            print(f"\n--- Example {i} ---")
            print(f"Before: {row[args.column][:100]}...")
            print(f"After:  {row[output_col][:100]}...")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60 + "\n")
    
    return all_stats


if __name__ == '__main__':
    main()
