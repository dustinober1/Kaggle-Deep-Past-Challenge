"""
Advanced Translation Extractor from Publications
=================================================
Extracts aligned translation pairs from OCR'd scholarly publications.
Uses more sophisticated patterns to identify Akkadian-English pairs.
"""

import pandas as pd
import re
import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "processed"

def is_akkadian_text(text):
    """Check if text looks like Akkadian transliteration."""
    # Akkadian markers
    akkadian_patterns = [
        r'\b[a-z]+-[a-z]+\b',  # Hyphenated syllables
        r'\b[A-Z]+\.[A-Z]+\b',  # Sumerian logograms with dots
        r'(?:um-ma|a-na|i-na|ša|šu-ma)\b',  # Common Akkadian words
        r'[àáèéìíùú]',  # Accented vowels
        r'[šṣṭḫ]',  # Special characters
    ]
    
    score = 0
    for pattern in akkadian_patterns:
        if re.search(pattern, text):
            score += 1
    
    # Must have at least 2 Akkadian markers and hyphenated words
    has_hyphens = len(re.findall(r'\b[a-z]+-[a-z]+\b', text)) >= 2
    return score >= 2 and has_hyphens

def is_english_translation(text):
    """Check if text looks like English translation."""
    # Common translation words
    english_markers = [
        r'\b(the|and|to|of|for|he|she|you|his|her|my|your|will|has|have|said)\b',
        r'\b(silver|minas|shekels|textiles|merchant|house|tablet)\b',
        r'\b(send|give|pay|bring|take|owe|receive)\b',
    ]
    
    score = 0
    text_lower = text.lower()
    for pattern in english_markers:
        matches = len(re.findall(pattern, text_lower))
        if matches > 0:
            score += matches
    
    # Should have several English words and no heavy Akkadian markers
    return score >= 3 and not is_akkadian_text(text)

def extract_translation_pairs(text, pdf_name, page):
    """
    Extract translation pairs from publication text.
    Look for patterns like:
    - Akkadian text followed by translation in quotes
    - Numbered lines with Akkadian and translation
    - Explicit "Translation:" sections
    """
    pairs = []
    
    # Pattern 1: Akkadian followed by quoted English
    # e.g., a-na pu-šu-kēn "to Puzur-kēn"
    pattern1 = r'([a-zšṣṭḫàáèéìíùú-]+(?:\s+[a-zšṣṭḫàáèéìíùú-]+){2,})\s*["""]([^"""]+)["""]'
    for match in re.finditer(pattern1, text):
        akk = match.group(1).strip()
        eng = match.group(2).strip()
        if is_akkadian_text(akk) and is_english_translation(eng) and len(eng) > 10:
            pairs.append({
                'transliteration': akk,
                'translation': eng,
                'source': 'publication_quoted',
                'pdf_name': pdf_name,
                'page': page
            })
    
    # Pattern 2: Line numbers with transliteration and translation
    # e.g., 1-3: um-ma pu-šu-kēn-ma: "Say to Puzur-kēn:"
    pattern2 = r'(\d+[-–]\d+|\d+)[:\s]+([a-zšṣṭḫàáèéìíùú][a-zšṣṭḫàáèéìíùú\s-]+)\s*[:=]\s*["""]?([^"""]+)["""]?'
    for match in re.finditer(pattern2, text):
        akk = match.group(2).strip()
        eng = match.group(3).strip()
        if is_akkadian_text(akk) and is_english_translation(eng) and len(eng) > 10 and len(eng) < 500:
            pairs.append({
                'transliteration': akk,
                'translation': eng,
                'source': 'publication_numbered',
                'pdf_name': pdf_name,
                'page': page
            })
    
    # Pattern 3: Parenthetical translations
    # e.g., um-ma a-ta-ma (thus you [said])
    pattern3 = r'([a-zšṣṭḫàáèéìíùú][a-zšṣṭḫàáèéìíùú\s-]{5,})\s*\(([^)]{10,200})\)'
    for match in re.finditer(pattern3, text):
        akk = match.group(1).strip()
        eng = match.group(2).strip()
        if is_akkadian_text(akk) and is_english_translation(eng):
            pairs.append({
                'transliteration': akk,
                'translation': eng,
                'source': 'publication_parenthetical',
                'pdf_name': pdf_name,
                'page': page
            })
    
    return pairs

def extract_all_translations():
    """Extract translations from all publications."""
    print("="*70)
    print("ADVANCED TRANSLATION EXTRACTION FROM PUBLICATIONS")
    print("="*70)
    
    # Load publications
    pub_df = pd.read_csv(DATA_DIR / "publications.csv", low_memory=False)
    print(f"Total pages: {len(pub_df)}")
    
    # Filter to pages with Akkadian
    akk_pages = pub_df[pub_df['has_akkadian'] == True].copy()
    print(f"Pages with Akkadian: {len(akk_pages)}")
    
    all_pairs = []
    
    for _, row in akk_pages.iterrows():
        text = str(row['page_text'])
        pdf_name = row['pdf_name']
        page = row['page']
        
        pairs = extract_translation_pairs(text, pdf_name, page)
        all_pairs.extend(pairs)
    
    print(f"\nExtracted pairs: {len(all_pairs)}")
    
    # Deduplicate by transliteration
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = pair['transliteration'].lower()
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
    
    print(f"Unique pairs: {len(unique_pairs)}")
    
    # Save
    if unique_pairs:
        pairs_df = pd.DataFrame(unique_pairs)
        pairs_df.to_csv(OUTPUT_DIR / "publication_extracted_pairs.csv", index=False)
        print(f"Saved to: {OUTPUT_DIR / 'publication_extracted_pairs.csv'}")
        
        # Show samples
        print("\nSample extracted pairs:")
        print("-"*50)
        for pair in unique_pairs[:5]:
            print(f"AKK: {pair['transliteration'][:60]}...")
            print(f"ENG: {pair['translation'][:60]}...")
            print()
    
    return unique_pairs

def create_final_training_set():
    """Create the final consolidated training set."""
    print("\n" + "="*70)
    print("CREATING FINAL TRAINING SET")
    print("="*70)
    
    all_data = []
    
    # 1. Original training data (document level)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    for _, row in train_df.iterrows():
        all_data.append({
            'transliteration': row['transliteration'],
            'translation': row['translation'],
            'source': 'train_document',
            'oare_id': row['oare_id']
        })
    print(f"Original training docs: {len(train_df)}")
    
    # 2. Aligned sentence pairs
    aligned_path = OUTPUT_DIR / "aligned_sentence_pairs.csv"
    if aligned_path.exists():
        aligned_df = pd.read_csv(aligned_path)
        for _, row in aligned_df.iterrows():
            if pd.notna(row['transliteration']) and pd.notna(row['translation']):
                all_data.append({
                    'transliteration': row['transliteration'],
                    'translation': row['translation'],
                    'source': 'aligned_sentence',
                    'oare_id': row.get('oare_id', '')
                })
        print(f"Aligned sentences: {len(aligned_df)}")
    
    # 3. Publication extracted pairs
    pub_path = OUTPUT_DIR / "publication_extracted_pairs.csv"
    if pub_path.exists():
        pub_df = pd.read_csv(pub_path)
        for _, row in pub_df.iterrows():
            all_data.append({
                'transliteration': row['transliteration'],
                'translation': row['translation'],
                'source': 'publication',
                'oare_id': ''
            })
        print(f"Publication pairs: {len(pub_df)}")
    
    # Create DataFrame
    final_df = pd.DataFrame(all_data)
    
    # Remove duplicates
    original_len = len(final_df)
    final_df = final_df.drop_duplicates(subset=['translation'], keep='first')
    print(f"Removed {original_len - len(final_df)} duplicates")
    
    # Filter valid pairs
    final_df = final_df[
        (final_df['transliteration'].str.len() > 10) &
        (final_df['translation'].str.len() > 5)
    ]
    
    # Save
    final_df.to_csv(OUTPUT_DIR / "final_training_data.csv", index=False)
    
    print(f"\nFinal training set: {len(final_df)}")
    print(f"  - Document level: {len(final_df[final_df['source'] == 'train_document'])}")
    print(f"  - Sentence level: {len(final_df[final_df['source'] == 'aligned_sentence'])}")
    print(f"  - Publications: {len(final_df[final_df['source'] == 'publication'])}")
    print(f"\nSaved to: {OUTPUT_DIR / 'final_training_data.csv'}")
    
    return final_df

if __name__ == "__main__":
    pairs = extract_all_translations()
    final = create_final_training_set()
