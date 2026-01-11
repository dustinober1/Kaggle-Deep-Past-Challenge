#!/usr/bin/env python3
"""
Lexicon Enhancement Pipeline for Deep Past Challenge

Creates word-level translation memory from eBL Dictionary and OA Lexicon:
1. word_dictionary.json - word → definition mappings
2. form_to_lexeme.json - transliteration form → normalized form → lexeme
3. named_entities.json - Personal Names (PN) and Geographic Names (GN)

Usage:
    python scripts/lexicon_builder.py [--verbose]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def clean_definition(definition: str) -> str:
    """Clean dictionary definition for use in translation memory."""
    if pd.isna(definition):
        return ""
    
    # Remove references like (cf. GAG §67), (GAG p.251)
    cleaned = re.sub(r'\(cf\.[^)]+\)', '', definition)
    cleaned = re.sub(r'\(GAG[^)]+\)', '', definition)
    
    # Remove numbered annotations like "2. vent. affix"
    cleaned = re.sub(r'\d+\.\s+[^;]+;?', '', cleaned)
    
    # Clean up quotes and whitespace
    cleaned = cleaned.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Extract core meaning (first quoted part if present)
    quote_match = re.search(r'"([^"]+)"', cleaned)
    if quote_match:
        return quote_match.group(1)
    
    return cleaned


def build_word_dictionary(dict_df: pd.DataFrame) -> dict:
    """Build word → definition mapping from eBL Dictionary."""
    word_dict = {}
    
    for _, row in dict_df.iterrows():
        word = row['word']
        definition = row['definition']
        
        if pd.isna(word) or pd.isna(definition):
            continue
        
        # Clean the word (remove Roman numerals for homonyms)
        clean_word = re.sub(r'\s+[IVX]+$', '', word.strip())
        
        cleaned_def = clean_definition(definition)
        if cleaned_def:
            # Keep first definition if word appears multiple times
            if clean_word not in word_dict:
                word_dict[clean_word] = cleaned_def
    
    return word_dict


def build_form_to_lexeme(lex_df: pd.DataFrame) -> dict:
    """Build form → norm → lexeme chain for regular words."""
    # Filter to words only (exclude PN and GN)
    words = lex_df[lex_df['type'] == 'word'].copy()
    
    form_map = {}
    
    for _, row in words.iterrows():
        form = row['form']
        norm = row['norm']
        lexeme = row['lexeme']
        
        if pd.isna(form):
            continue
        
        # Normalize form for lookup (lowercase, strip)
        form_key = form.strip().lower()
        
        entry = {
            'form': form,
            'norm': norm if pd.notna(norm) else form,
            'lexeme': lexeme if pd.notna(lexeme) else (norm if pd.notna(norm) else form)
        }
        
        # Keep first mapping for each form
        if form_key not in form_map:
            form_map[form_key] = entry
    
    return form_map


def build_named_entities(lex_df: pd.DataFrame) -> dict:
    """Build separate indices for Personal Names and Geographic Names."""
    entities = {
        'PN': [],  # Personal Names
        'GN': []   # Geographic Names
    }
    
    for entity_type in ['PN', 'GN']:
        subset = lex_df[lex_df['type'] == entity_type]
        
        for _, row in subset.iterrows():
            form = row['form']
            norm = row['norm']
            
            if pd.isna(form):
                continue
            
            # Check if it's a female name
            is_female = pd.notna(row.get('Female(f)')) and row['Female(f)'] == 'f'
            
            entry = {
                'form': form,
                'norm': norm if pd.notna(norm) else form,
            }
            
            if entity_type == 'PN' and is_female:
                entry['gender'] = 'female'
            
            entities[entity_type].append(entry)
    
    # Create lookup sets for quick matching
    entities['PN_forms'] = list(set(e['form'].lower() for e in entities['PN']))
    entities['GN_forms'] = list(set(e['form'].lower() for e in entities['GN']))
    
    return entities


def compute_coverage(form_map: dict, train_df: pd.DataFrame) -> dict:
    """Compute coverage statistics on training data."""
    all_tokens = set()
    matched_tokens = set()
    
    for text in train_df['transliteration'].dropna():
        # Tokenize: split on whitespace and hyphens
        tokens = text.lower().replace('-', ' ').split()
        all_tokens.update(tokens)
        
        for token in tokens:
            if token in form_map:
                matched_tokens.add(token)
    
    return {
        'total_tokens': len(all_tokens),
        'matched_tokens': len(matched_tokens),
        'coverage_pct': 100 * len(matched_tokens) / len(all_tokens) if all_tokens else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Build lexicon lookup tables')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed output')
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    if args.verbose:
        print("Loading eBL Dictionary...")
    dict_df = pd.read_csv(data_dir / 'eBL_Dictionary.csv')
    
    if args.verbose:
        print("Loading OA Lexicon...")
    lex_df = pd.read_csv(data_dir / 'OA_Lexicon_eBL.csv')
    
    # Build resources
    if args.verbose:
        print("\nBuilding word dictionary...")
    word_dict = build_word_dictionary(dict_df)
    if args.verbose:
        print(f"  → {len(word_dict)} words with definitions")
    
    if args.verbose:
        print("\nBuilding form-to-lexeme mapping...")
    form_map = build_form_to_lexeme(lex_df)
    if args.verbose:
        print(f"  → {len(form_map)} form mappings")
    
    if args.verbose:
        print("\nBuilding named entity index...")
    entities = build_named_entities(lex_df)
    if args.verbose:
        print(f"  → {len(entities['PN'])} personal names")
        print(f"  → {len(entities['GN'])} geographic names")
    
    # Compute coverage on training data
    if args.verbose:
        print("\nComputing coverage on training data...")
    try:
        train_df = pd.read_csv(output_dir / 'augmented_train.csv')
        coverage = compute_coverage(form_map, train_df)
        if args.verbose:
            print(f"  → {coverage['matched_tokens']}/{coverage['total_tokens']} tokens matched")
            print(f"  → {coverage['coverage_pct']:.1f}% coverage")
    except FileNotFoundError:
        coverage = None
        if args.verbose:
            print("  → Training data not found, skipping coverage")
    
    # Save outputs
    if args.verbose:
        print("\nSaving output files...")
    
    with open(output_dir / 'word_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(word_dict, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / 'form_to_lexeme.json', 'w', encoding='utf-8') as f:
        json.dump(form_map, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / 'named_entities.json', 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n=== Lexicon Enhancement Complete ===")
    print(f"Word dictionary:     {len(word_dict):,} entries → word_dictionary.json")
    print(f"Form-to-lexeme:      {len(form_map):,} entries → form_to_lexeme.json")
    print(f"Named entities (PN): {len(entities['PN']):,} entries")
    print(f"Named entities (GN): {len(entities['GN']):,} entries → named_entities.json")
    if coverage:
        print(f"Training coverage:   {coverage['coverage_pct']:.1f}%")
    
    # Sample lookups for verification
    if args.verbose:
        print("\n=== Sample Lookups ===")
        
        # Dictionary samples
        sample_words = ['ana', 'šarru', 'bēlu', 'nadānu', 'amāru']
        print("\nWord definitions:")
        for w in sample_words:
            if w in word_dict:
                print(f"  {w} → {word_dict[w]}")
        
        # Form samples
        sample_forms = ['a-na', 'kù.babbar', 'dumu', 'ma-na']
        print("\nForm to lexeme:")
        for f in sample_forms:
            if f in form_map:
                entry = form_map[f]
                print(f"  {f} → {entry['norm']} → {entry['lexeme']}")


if __name__ == '__main__':
    main()
