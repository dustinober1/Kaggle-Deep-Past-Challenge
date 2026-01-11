#!/usr/bin/env python3
"""
Special Token Handler for Deep Past Challenge

This script handles special tokens in Akkadian transliterations:
1. Sumerian logograms (ALL CAPS) - builds vocabulary
2. Subscript/superscript numbers normalization
3. Personal names (<PN>) tokenization
4. Geographic names (<GN>) tokenization  
5. Numbers and measurements normalization

Usage:
    python scripts/special_token_handler.py --input FILE --output FILE [--options]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

# Common Sumerian logograms in Old Assyrian texts
SUMERIAN_LOGOGRAMS = {
    # Metals and materials
    'KÙ.BABBAR': 'silver',
    'KÙ.GI': 'gold',
    'URUDU': 'copper',
    'AN.NA': 'tin',
    'ZABAR': 'bronze',
    'NA₄': 'stone',
    
    # Units of measurement
    'GÍN': 'shekel',
    'GÚ': 'talent',
    'SÌLA': 'liter',
    'BÁN': 'measure',
    
    # People and professions
    'DUMU': 'son',
    'DAM': 'wife',
    'LUGAL': 'king',
    'DAM.GÀR': 'merchant',
    'LÚ': 'man',
    'MUNUS': 'woman',
    
    # Objects and items
    'TÚG': 'textile',
    'ANŠE': 'donkey',
    'GIŠ': 'wood',
    'É': 'house',
    'É.GAL': 'palace',
    
    # Documents
    'KIŠIB': 'seal',
    'DUB': 'tablet',
    
    # Deities
    'DINGIR': 'god',
    'IŠTAR': 'Ištar',
    'UTU': 'Šamaš',
    'IM': 'Adad',
    'EN.ZU': 'Sin',
    'EN.LÍL': 'Enlil',
    'IŠKUR': 'Adad',
    
    # Time
    'ITU': 'month',
    'ITU.KAM': 'month',
    'U₄': 'day',
    'MU': 'year',
    
    # Other common logograms
    'IGI': 'before/witness',
    'GÌR': 'dagger',
    'ŠU': 'hand',
    'ŠÀ': 'heart',
    'ŠÀ.BA': 'thereof',
    'ŠU.NÍGIN': 'total',
    'SIG₅': 'good/fine',
    'ḪI.A': 'plural',
    'MEŠ': 'plural',
}

# Subscript number mapping
SUBSCRIPT_MAP = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
}

# Reverse mapping for normalization
NORMAL_TO_SUBSCRIPT = {v: k for k, v in SUBSCRIPT_MAP.items()}

# Measurement units (used with numbers)
MEASUREMENT_UNITS = {
    'ma-na': 'mina',
    'GÍN': 'shekel', 
    'GÚ': 'talent',
    'SÌLA': 'liter',
}


# =============================================================================
# LOGOGRAM FUNCTIONS
# =============================================================================

def extract_logograms(text: str) -> List[str]:
    """
    Extract Sumerian logograms (ALL CAPS sequences) from text.
    
    Logograms are typically:
    - All uppercase letters (including Š, Ṣ, Ṭ, Ḫ)
    - May include dots (KÙ.BABBAR)
    - May include subscript numbers (SIG₅)
    """
    if not text or pd.isna(text):
        return []
    
    # Pattern for Sumerian logograms
    # Match sequences of uppercase + special chars + dots + subscripts
    pattern = r'\b([A-ZŠṢṬḪ][A-ZŠṢṬḪ₀₁₂₃₄₅₆₇₈₉\.]+)\b'
    
    matches = re.findall(pattern, text)
    
    # Filter out single letters and clean up
    logograms = [m for m in matches if len(m) > 1]
    
    return logograms


def build_logogram_vocabulary(texts: List[str]) -> Dict[str, int]:
    """
    Build vocabulary of all Sumerian logograms found in texts.
    
    Returns dict of logogram → count.
    """
    counter = Counter()
    
    for text in texts:
        logograms = extract_logograms(text)
        counter.update(logograms)
    
    return dict(counter.most_common())


def wrap_logograms(text: str, marker: str = 'LOG') -> str:
    """
    Wrap Sumerian logograms with markers for visibility.
    
    Example: KÙ.BABBAR → <LOG>KÙ.BABBAR</LOG>
    
    Note: This is optional - logograms may be left as-is.
    """
    if not text or pd.isna(text):
        return ""
    
    pattern = r'\b([A-ZŠṢṬḪ][A-ZŠṢṬḪ₀₁₂₃₄₅₆₇₈₉\.]+)\b'
    
    def replace_func(match):
        logo = match.group(1)
        if len(logo) > 1:
            return f'<{marker}>{logo}</{marker}>'
        return logo
    
    return re.sub(pattern, replace_func, text)


# =============================================================================
# SUBSCRIPT NUMBER FUNCTIONS
# =============================================================================

def normalize_subscripts(text: str, to_normal: bool = True) -> str:
    """
    Normalize subscript numbers.
    
    Args:
        text: Input text
        to_normal: If True, convert ₂ → 2. If False, convert 2 → ₂
    """
    if not text or pd.isna(text):
        return ""
    
    mapping = SUBSCRIPT_MAP if to_normal else NORMAL_TO_SUBSCRIPT
    
    for src, dst in mapping.items():
        text = text.replace(src, dst)
    
    return text


def extract_subscript_patterns(text: str) -> List[str]:
    """
    Extract patterns containing subscript numbers.
    
    Returns list of tokens with subscripts (e.g., 'qí-bi₄-ma', 'il₅-qé')
    """
    if not text or pd.isna(text):
        return []
    
    # Pattern for words containing subscripts
    pattern = r'\b[\w-]*[₀₁₂₃₄₅₆₇₈₉]+[\w-]*\b'
    
    return re.findall(pattern, text)


# =============================================================================
# NAMED ENTITY FUNCTIONS
# =============================================================================

def load_named_entities(filepath: Path) -> Tuple[Set[str], Set[str]]:
    """
    Load personal names (PN) and geographic names (GN) from JSON file.
    
    Returns (pn_set, gn_set) of normalized forms.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pn_forms = set()
    gn_forms = set()
    
    # Extract PN forms
    if 'PN' in data:
        for entry in data['PN']:
            if 'form' in entry:
                pn_forms.add(entry['form'].lower())
    
    # Extract GN forms  
    if 'GN' in data:
        for entry in data['GN']:
            if 'form' in entry:
                gn_forms.add(entry['form'].lower())
    
    return pn_forms, gn_forms


def identify_names_after_dumu(text: str) -> List[Tuple[str, int, int]]:
    """
    Identify likely personal names following DUMU (son of).
    
    Returns list of (name, start, end) tuples.
    """
    if not text or pd.isna(text):
        return []
    
    names = []
    
    # Pattern: DUMU followed by a name (hyphenated word)
    # Example: DUMU ṣí-lá-{d}IM
    pattern = r'DUMU\s+([a-zšṣṭḫáàâäéèêëíìîïóòôöúùûü₀₁₂₃₄₅₆₇₈₉\-\{\}]+)'
    
    for match in re.finditer(pattern, text, re.IGNORECASE):
        names.append((match.group(1), match.start(1), match.end(1)))
    
    return names


def replace_names_with_tokens(text: str, pn_forms: Set[str], gn_forms: Set[str],
                              use_tokens: bool = False) -> str:
    """
    Optionally replace personal/geographic names with <PN>/<GN> tokens.
    
    Note: This is aggressive and may lose information. Use with caution.
    By default, names are NOT replaced (use_tokens=False).
    """
    if not text or pd.isna(text) or not use_tokens:
        return text
    
    words = text.split()
    result = []
    
    for word in words:
        word_lower = word.lower()
        
        # Check if word matches a known PN
        if word_lower in pn_forms:
            result.append('<PN>')
        # Check if word matches a known GN
        elif word_lower in gn_forms:
            result.append('<GN>')
        else:
            result.append(word)
    
    return ' '.join(result)


# =============================================================================
# NUMBER AND MEASUREMENT FUNCTIONS
# =============================================================================

def extract_numbers(text: str) -> List[str]:
    """
    Extract numeric patterns from text.
    
    Patterns include:
    - Integers: 10, 17, 100
    - Decimals: 0.33333, 1.5, 27.83333
    - Numbers with units: 10 ma-na, 2 GÍN
    """
    if not text or pd.isna(text):
        return []
    
    # Pattern for numbers (including decimals)
    pattern = r'\b\d+(?:\.\d+)?\b'
    
    return re.findall(pattern, text)


def extract_measurements(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract number + unit patterns.
    
    Returns list of (number, unit, full_match) tuples.
    """
    if not text or pd.isna(text):
        return []
    
    measurements = []
    
    # Pattern: number followed by unit
    pattern = r'(\d+(?:\.\d+)?)\s*(ma-na|GÍN|GÚ|SÌLA|GÍN\.TA)'
    
    for match in re.finditer(pattern, text):
        measurements.append((match.group(1), match.group(2), match.group(0)))
    
    return measurements


def normalize_numbers(text: str, use_token: bool = False) -> str:
    """
    Optionally normalize numbers to <NUM> token.
    
    Args:
        text: Input text
        use_token: If True, replace numbers with <NUM>. If False, keep as-is.
    """
    if not text or pd.isna(text) or not use_token:
        return text
    
    # Replace standalone numbers
    text = re.sub(r'\b\d+(?:\.\d+)?\b', '<NUM>', text)
    
    return text


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_text(text: str, 
                 pn_forms: Optional[Set[str]] = None,
                 gn_forms: Optional[Set[str]] = None,
                 wrap_logos: bool = False,
                 replace_names: bool = False,
                 normalize_nums: bool = False) -> str:
    """
    Apply special token processing to text.
    
    Args:
        text: Input transliteration
        pn_forms: Set of personal name forms
        gn_forms: Set of geographic name forms
        wrap_logos: If True, wrap logograms with <LOG> tags
        replace_names: If True, replace names with <PN>/<GN>
        normalize_nums: If True, replace numbers with <NUM>
        
    Returns:
        Processed text
    """
    if not text or pd.isna(text):
        return ""
    
    # Step 1: Optionally wrap logograms
    if wrap_logos:
        text = wrap_logograms(text)
    
    # Step 2: Optionally replace names with tokens
    if replace_names and pn_forms and gn_forms:
        text = replace_names_with_tokens(text, pn_forms, gn_forms, use_tokens=True)
    
    # Step 3: Optionally normalize numbers
    if normalize_nums:
        text = normalize_numbers(text, use_token=True)
    
    return text


def analyze_dataset(df: pd.DataFrame, 
                    text_col: str = 'transliteration_cleaned',
                    entities_path: Optional[Path] = None) -> Dict:
    """
    Analyze dataset for special tokens and patterns.
    
    Returns statistics dict.
    """
    texts = df[text_col].dropna().tolist()
    
    # Build logogram vocabulary
    logogram_vocab = build_logogram_vocabulary(texts)
    
    # Extract all subscript patterns
    subscript_patterns = Counter()
    for text in texts:
        patterns = extract_subscript_patterns(text)
        subscript_patterns.update(patterns)
    
    # Extract all numbers
    all_numbers = Counter()
    for text in texts:
        numbers = extract_numbers(text)
        all_numbers.update(numbers)
    
    # Extract measurements
    all_measurements = Counter()
    for text in texts:
        measurements = extract_measurements(text)
        for num, unit, full in measurements:
            all_measurements[full] += 1
    
    # Load named entities if available
    pn_count = 0
    gn_count = 0
    if entities_path and entities_path.exists():
        pn_forms, gn_forms = load_named_entities(entities_path)
        pn_count = len(pn_forms)
        gn_count = len(gn_forms)
    
    stats = {
        'total_texts': len(texts),
        'logogram_vocabulary_size': len(logogram_vocab),
        'top_logograms': dict(Counter(logogram_vocab).most_common(30)),
        'subscript_patterns_count': len(subscript_patterns),
        'top_subscript_patterns': dict(subscript_patterns.most_common(20)),
        'unique_numbers': len(all_numbers),
        'top_numbers': dict(all_numbers.most_common(20)),
        'measurement_patterns': dict(all_measurements.most_common(20)),
        'pn_forms_count': pn_count,
        'gn_forms_count': gn_count,
    }
    
    return stats, logogram_vocab


def save_vocabularies(logogram_vocab: Dict[str, int], 
                     output_dir: Path,
                     pn_forms: Optional[Set[str]] = None,
                     gn_forms: Optional[Set[str]] = None):
    """
    Save vocabulary files for tokenizer training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save logogram vocabulary
    logo_path = output_dir / 'logogram_vocabulary.json'
    with open(logo_path, 'w', encoding='utf-8') as f:
        json.dump(logogram_vocab, f, ensure_ascii=False, indent=2)
    print(f"  Saved logogram vocabulary: {logo_path}")
    
    # Save as simple word list too
    logo_list_path = output_dir / 'logograms.txt'
    with open(logo_list_path, 'w', encoding='utf-8') as f:
        for logo in logogram_vocab.keys():
            f.write(f"{logo}\n")
    print(f"  Saved logogram list: {logo_list_path}")
    
    # Save known Sumerian meanings
    meanings_path = output_dir / 'logogram_meanings.json'
    with open(meanings_path, 'w', encoding='utf-8') as f:
        json.dump(SUMERIAN_LOGOGRAMS, f, ensure_ascii=False, indent=2)
    print(f"  Saved logogram meanings: {meanings_path}")
    
    # Save PN list if available
    if pn_forms:
        pn_path = output_dir / 'personal_names.txt'
        with open(pn_path, 'w', encoding='utf-8') as f:
            for name in sorted(pn_forms):
                f.write(f"{name}\n")
        print(f"  Saved personal names: {pn_path}")
    
    # Save GN list if available
    if gn_forms:
        gn_path = output_dir / 'geographic_names.txt'
        with open(gn_path, 'w', encoding='utf-8') as f:
            for name in sorted(gn_forms):
                f.write(f"{name}\n")
        print(f"  Saved geographic names: {gn_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Special Token Handler for Deep Past Challenge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', type=str, 
                       default='data/processed/train_cleaned.csv',
                       help='Input CSV file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output CSV file (optional)')
    parser.add_argument('--column', '-c', type=str, 
                       default='transliteration_cleaned',
                       help='Column to process')
    parser.add_argument('--entities', '-e', type=str,
                       default='data/processed/named_entities.json',
                       help='Named entities JSON file')
    parser.add_argument('--vocab-dir', type=str,
                       default='models/tokenizers',
                       help='Directory to save vocabularies')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze, do not modify data')
    parser.add_argument('--wrap-logograms', action='store_true',
                       help='Wrap logograms with <LOG> tags')
    parser.add_argument('--replace-names', action='store_true',
                       help='Replace names with <PN>/<GN> tokens')
    parser.add_argument('--normalize-numbers', action='store_true',
                       help='Replace numbers with <NUM> token')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    entities_path = project_root / args.entities
    vocab_dir = project_root / args.vocab_dir
    
    print("=" * 60)
    print("SPECIAL TOKEN HANDLER")
    print("Phase 3.2 - Deep Past Challenge")
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
    
    # Load named entities
    pn_forms, gn_forms = None, None
    if entities_path.exists():
        print(f"\nLoading named entities: {entities_path}")
        pn_forms, gn_forms = load_named_entities(entities_path)
        print(f"  Personal names (PN): {len(pn_forms):,}")
        print(f"  Geographic names (GN): {len(gn_forms):,}")
    
    # Analyze dataset
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    stats, logogram_vocab = analyze_dataset(df, args.column, entities_path)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total texts: {stats['total_texts']:,}")
    print(f"  Unique logograms: {stats['logogram_vocabulary_size']:,}")
    print(f"  Subscript patterns: {stats['subscript_patterns_count']:,}")
    print(f"  Unique numbers: {stats['unique_numbers']}")
    
    print(f"\n📝 Top 15 Sumerian Logograms:")
    for i, (logo, count) in enumerate(list(stats['top_logograms'].items())[:15], 1):
        meaning = SUMERIAN_LOGOGRAMS.get(logo, '?')
        print(f"  {i:2}. {logo:15} ({count:,}) - {meaning}")
    
    print(f"\n🔢 Top 10 Subscript Patterns:")
    for i, (pattern, count) in enumerate(list(stats['top_subscript_patterns'].items())[:10], 1):
        print(f"  {i:2}. {pattern:20} ({count:,})")
    
    print(f"\n📏 Top 10 Measurements:")
    for i, (meas, count) in enumerate(list(stats['measurement_patterns'].items())[:10], 1):
        print(f"  {i:2}. {meas:20} ({count:,})")
    
    # Save vocabularies
    print("\n" + "=" * 60)
    print("SAVING VOCABULARIES")
    print("=" * 60)
    
    save_vocabularies(logogram_vocab, vocab_dir, pn_forms, gn_forms)
    
    # Optionally process and save modified data
    if not args.analyze_only and args.output:
        print("\n" + "=" * 60)
        print("PROCESSING DATA")
        print("=" * 60)
        
        output_path = project_root / args.output
        output_col = f"{args.column}_processed"
        
        df[output_col] = df[args.column].apply(
            lambda x: process_text(
                x, 
                pn_forms=pn_forms,
                gn_forms=gn_forms,
                wrap_logos=args.wrap_logograms,
                replace_names=args.replace_names,
                normalize_nums=args.normalize_numbers
            )
        )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to: {output_path}")
    
    # Save analysis stats
    stats_path = vocab_dir / 'special_token_stats.json'
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved statistics: {stats_path}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60 + "\n")
    
    return stats, logogram_vocab


if __name__ == '__main__':
    main()
