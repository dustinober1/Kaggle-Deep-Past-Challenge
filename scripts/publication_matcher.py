#!/usr/bin/env python3
"""
Publication Matcher - Phase 1.2 Step 1

Links published_texts.csv aliases to publications.csv pdf_names by parsing
publication series codes (AKT, CCT, BIN, ICK, etc.) from both sources.

Output: data/processed/publication_links.csv
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Known publication series patterns found in both datasets
PUBLICATION_SERIES = {
    'AKT': r'AKT\s*(\d+[a-z]?)\s*,?\s*(\d+)?',      # AKT 1, 49 or AKT 1 49
    'CCT': r'CCT\s*(\d+)\s+(\d+[a-z]?)',             # CCT 6 17a
    'BIN': r'BIN\s*(\d+)\s+(\d+)',                   # BIN 4 112
    'ICK': r'ICK\s*(\d+)\s+(\d+)',                   # ICK 1 146
    'TC': r'TC\s*(\d+)\s+(\d+)',                     # TC 3 1
    'TPAK': r'TPAK\s*(\d+)\s+(\d+[a-z]?)',           # TPAK 1 79b
    'Prag': r'Prag\s*I?\s*(\d+[a-z]?)',              # Prag I 451a
    'RA': r'RA\s*(\d+)\s+(\d+)',                     # RA 60 125
    'Or': r'Or\s*(\d+)\s+(\d+)',                     # Or 36 396
    'JCS': r'JCS\s*(\d+)',                           # JCS 59
    'AfO': r'AfO\s*(\d+)',                           # AfO 51
    'AnSt': r'AnSt\s*(\d+)',                         # AnSt 45
    'ZA': r'ZA\s*(\d+)',                             # ZA 75
}


def extract_series_from_alias(alias: str) -> List[Dict]:
    """Extract publication series references from an alias string."""
    if pd.isna(alias):
        return []
    
    results = []
    # Split on pipe for multiple aliases
    parts = str(alias).split('|')
    
    for part in parts:
        part = part.strip()
        for series, pattern in PUBLICATION_SERIES.items():
            match = re.search(pattern, part, re.IGNORECASE)
            if match:
                groups = match.groups()
                results.append({
                    'series': series,
                    'volume': groups[0] if groups else None,
                    'number': groups[1] if len(groups) > 1 else None,
                    'raw': part
                })
    
    return results


def extract_series_from_pdf_name(pdf_name: str) -> Optional[Dict]:
    """Extract publication series from PDF filename."""
    if pd.isna(pdf_name):
        return None
    
    pdf_name = str(pdf_name)
    
    # Common patterns in PDF names
    patterns = [
        (r'AKT\s*(\d+[a-z]?)', 'AKT'),
        (r'CCT\s*(\d+)', 'CCT'),
        (r'BIN\s*(\d+)', 'BIN'),
        (r'ICK\s*(\d+)', 'ICK'),
        (r'TC\s*(\d+)', 'TC'),
        (r'TPAK\s*(\d+)', 'TPAK'),
        (r'Prag\s*I?', 'Prag'),
        (r'RA\s*(\d+)', 'RA'),
        (r'Or\s*(\d+)', 'Or'),
        (r'JCS\s*(\d+)', 'JCS'),
        (r'AfO\s*(\d+)', 'AfO'),
        (r'AnSt\s*(\d+)', 'AnSt'),
        (r'ZA\s*(\d+)', 'ZA'),
    ]
    
    for pattern, series in patterns:
        match = re.search(pattern, pdf_name, re.IGNORECASE)
        if match:
            return {
                'series': series,
                'volume': match.group(1) if match.groups() else None,
                'pdf_name': pdf_name
            }
    
    return None


def match_publications(
    texts_df: pd.DataFrame,
    pubs_df: pd.DataFrame,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Match published_texts to publications via series/volume matching.
    
    Returns DataFrame with: oare_id, pdf_name, series, volume, match_confidence
    """
    # Build PDF index by series/volume
    pdf_index = {}
    for pdf_name in pubs_df['pdf_name'].unique():
        info = extract_series_from_pdf_name(pdf_name)
        if info:
            key = (info['series'], info['volume'])
            if key not in pdf_index:
                pdf_index[key] = []
            pdf_index[key].append(pdf_name)
    
    if verbose:
        print(f"Built PDF index with {len(pdf_index)} series/volume combinations")
        for (series, vol), pdfs in sorted(pdf_index.items())[:10]:
            print(f"  {series} {vol}: {len(pdfs)} PDFs")
    
    # Match each text
    matches = []
    for _, row in texts_df.iterrows():
        oare_id = row['oare_id']
        alias = row.get('aliases', '')
        pub_cat = row.get('publication_catalog', '')
        
        # Try both alias and publication_catalog
        for source_field, source_val in [('aliases', alias), ('publication_catalog', pub_cat)]:
            refs = extract_series_from_alias(source_val)
            
            for ref in refs:
                key = (ref['series'], ref['volume'])
                if key in pdf_index:
                    for pdf_name in pdf_index[key]:
                        matches.append({
                            'oare_id': oare_id,
                            'pdf_name': pdf_name,
                            'series': ref['series'],
                            'volume': ref['volume'],
                            'number': ref['number'],
                            'match_source': source_field,
                            'raw_reference': ref['raw']
                        })
    
    df = pd.DataFrame(matches)
    
    # Remove duplicates (same oare_id + pdf_name)
    if len(df) > 0:
        df = df.drop_duplicates(subset=['oare_id', 'pdf_name'])
    
    return df


def validate_matches(matches_df: pd.DataFrame, texts_df: pd.DataFrame, train_df: pd.DataFrame):
    """Print validation statistics for matches."""
    print("\n=== VALIDATION RESULTS ===\n")
    
    total_matches = len(matches_df)
    unique_texts = matches_df['oare_id'].nunique()
    unique_pdfs = matches_df['pdf_name'].nunique()
    
    print(f"Total matches: {total_matches}")
    print(f"Unique texts matched: {unique_texts}")
    print(f"Unique PDFs matched: {unique_pdfs}")
    
    # How many are NOT in train.csv?
    train_ids = set(train_df['oare_id'])
    new_texts = matches_df[~matches_df['oare_id'].isin(train_ids)]
    print(f"\nTexts NOT in train.csv: {new_texts['oare_id'].nunique()}")
    
    # By series
    print("\nMatches by series:")
    for series, group in matches_df.groupby('series'):
        unique_in_series = group['oare_id'].nunique()
        print(f"  {series}: {unique_in_series} texts")


def main():
    parser = argparse.ArgumentParser(description='Match published texts to PDF publications')
    parser.add_argument('--validate', action='store_true', help='Run validation after matching')
    parser.add_argument('--output', type=str, default='data/processed/publication_links.csv',
                        help='Output file path')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    texts_path = base_dir / 'data' / 'published_texts.csv'
    pubs_path = base_dir / 'data' / 'publications.csv'
    train_path = base_dir / 'data' / 'train.csv'
    output_path = base_dir / args.output
    
    print("Loading data...")
    texts_df = pd.read_csv(texts_path)
    pubs_df = pd.read_csv(pubs_path)
    train_df = pd.read_csv(train_path)
    
    print(f"Published texts: {len(texts_df)}")
    print(f"Publication pages: {len(pubs_df)} from {pubs_df['pdf_name'].nunique()} PDFs")
    print(f"Training samples: {len(train_df)}")
    
    print("\nMatching publications...")
    matches_df = match_publications(texts_df, pubs_df, verbose=True)
    
    if args.validate:
        validate_matches(matches_df, texts_df, train_df)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(matches_df)} matches to {output_path}")


if __name__ == '__main__':
    main()
