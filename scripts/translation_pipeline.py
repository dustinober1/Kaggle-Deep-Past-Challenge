#!/usr/bin/env python3
"""
Translation Pipeline - Phase 1.2 Step 3

Translates German and French content to English using Gemini.
Takes extracted_translations.csv and creates translated_content.csv.

Input: data/processed/extracted_translations.csv
Output: data/processed/translated_content.csv
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")


TRANSLATION_PROMPT = """Translate the following {source_lang} text about ancient Mesopotamian/Assyrian topics to English.

This is an academic translation from an Assyriology publication. Preserve:
- Technical terms and names
- Scholarly conventions
- Formal academic register

Text to translate:
{text}

English translation:"""

LANGUAGE_MAP = {
    'DE': 'German',
    'FR': 'French',
    'German': 'German',
    'French': 'French'
}


def setup_gemini(api_key: Optional[str] = None) -> bool:
    """Configure Gemini API."""
    if not GEMINI_AVAILABLE:
        return False
    
    key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not key:
        print("Error: No Gemini API key found. Set GEMINI_API_KEY environment variable.")
        return False
    
    genai.configure(api_key=key)
    return True


def translate_text(
    text: str,
    source_lang: str,
    model: "genai.GenerativeModel",
    max_retries: int = 3
) -> Optional[str]:
    """Translate text from source language to English."""
    if not text or pd.isna(text):
        return None
    
    lang_name = LANGUAGE_MAP.get(source_lang, source_lang)
    prompt = TRANSLATION_PROMPT.format(source_lang=lang_name, text=text)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(30)
                continue
            print(f"Translation error: {e}")
            return None
    
    return None


def process_translations(
    df: pd.DataFrame,
    model: "genai.GenerativeModel",
    dry_run: bool = False
) -> pd.DataFrame:
    """Translate non-English content to English."""
    
    # Filter for non-English entries
    non_english = df[df['language'].isin(['DE', 'FR', 'German', 'French'])].copy()
    english = df[df['language'].isin(['EN', 'English'])].copy()
    
    print(f"Total extractions: {len(df)}")
    print(f"  Already English: {len(english)}")
    print(f"  Need translation: {len(non_english)}")
    
    if dry_run:
        print("\nDry run - showing sample texts to translate:")
        for _, row in non_english.head(3).iterrows():
            print(f"\n  [{row['language']}] {row['translation'][:100]}...")
        return df
    
    # Translate non-English content
    translated_texts = []
    for i, (idx, row) in enumerate(non_english.iterrows()):
        if i % 10 == 0:
            print(f"Translating {i+1}/{len(non_english)}...")
        
        translated = translate_text(row['translation'], row['language'], model)
        translated_texts.append(translated)
        time.sleep(0.5)  # Rate limiting
    
    non_english['translation_en'] = translated_texts
    non_english['original_translation'] = non_english['translation']
    non_english['translation'] = non_english['translation_en']
    non_english['original_language'] = non_english['language']
    non_english['language'] = 'EN'
    
    # Combine with already-English content
    english['original_language'] = 'EN'
    english['original_translation'] = english['translation']
    english['translation_en'] = english['translation']
    
    result = pd.concat([english, non_english], ignore_index=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Translate DE/FR content to English')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be translated without API calls')
    parser.add_argument('--api-key', type=str, help='Gemini API key')
    parser.add_argument('--input', type=str, default='data/processed/extracted_translations.csv')
    parser.add_argument('--output', type=str, default='data/processed/translated_content.csv')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / args.input
    output_path = base_dir / args.output
    
    if not input_path.exists():
        print(f"Error: Run translation_extractor.py first to create {input_path}")
        return
    
    print("Loading extracted translations...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} extractions")
    
    if args.dry_run:
        process_translations(df, None, dry_run=True)
        return
    
    if not setup_gemini(args.api_key):
        return
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini API configured")
    
    result = process_translations(df, model)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved {len(result)} translations to {output_path}")


if __name__ == '__main__':
    main()
