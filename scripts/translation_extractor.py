#!/usr/bin/env python3
"""
Translation Extractor - Phase 1.2 Step 2

Uses Gemini LLM to extract translation sections from publication OCR text.
Identifies transliteration + translation pairs from matched PDF pages.

Input: publication_links.csv, publications.csv
Output: data/processed/extracted_translations.csv
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")


EXTRACTION_PROMPT = """You are an expert in Assyriology analyzing OCR text from academic publications about ancient Mesopotamian texts.

Given the following page content from a publication, identify if it contains translations of Akkadian/Old Assyrian texts.

For EACH translation found, extract:
1. The text identifier (if mentioned, e.g. "Text 1", "Kt a/k 123", etc.)
2. The translation (English, German, or French)
3. The language of the translation (EN, DE, or FR)

Return a JSON array of objects with these fields:
- "text_id": identifier if found, or null
- "translation": the extracted translation text
- "language": "EN", "DE", or "FR"
- "confidence": "high", "medium", or "low"

If there are NO translations found, return an empty array: []

PAGE CONTENT:
{page_text}

IMPORTANT:
- Only extract actual TRANSLATIONS, not transliterations (which use cuneiform sign names)
- Translations are in modern language (English, German, French) rendering the meaning
- Ignore bibliography, footnotes about grammar, and editorial comments
- Focus on complete sentences or meaningful phrases

JSON Response:"""


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


def extract_translations_from_page(
    page_text: str,
    model: "genai.GenerativeModel",
    max_retries: int = 3
) -> List[Dict]:
    """Extract translations from a single page using Gemini."""
    if not page_text or len(str(page_text).strip()) < 100:
        return []
    
    prompt = EXTRACTION_PROMPT.format(page_text=str(page_text)[:8000])  # Truncate long pages
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Extract JSON from response
            if text.startswith('['):
                json_str = text
            else:
                # Try to find JSON array in response
                match = re.search(r'\[[\s\S]*\]', text)
                if match:
                    json_str = match.group()
                else:
                    return []
            
            results = json.loads(json_str)
            return results if isinstance(results, list) else []
            
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                # Rate limit - wait and retry
                time.sleep(30)
                continue
            print(f"Error extracting: {e}")
            return []
    
    return []


def process_matches(
    links_df: pd.DataFrame,
    pubs_df: pd.DataFrame,
    model: "genai.GenerativeModel",
    sample_size: Optional[int] = None,
    dry_run: bool = False
) -> pd.DataFrame:
    """Process matched publications and extract translations."""
    
    # Get unique PDF names from links
    pdf_names = links_df['pdf_name'].unique()
    print(f"Processing {len(pdf_names)} unique PDFs...")
    
    if sample_size:
        pdf_names = pdf_names[:sample_size]
        print(f"  (limited to {sample_size} for sampling)")
    
    all_extractions = []
    
    for i, pdf_name in enumerate(pdf_names):
        print(f"\n[{i+1}/{len(pdf_names)}] {pdf_name[:60]}...")
        
        # Get pages from this PDF (include all pages since translations may 
        # appear without Akkadian markers)
        pdf_pages = pubs_df[pubs_df['pdf_name'] == pdf_name].head(50)  # Limit pages per PDF
        
        if len(pdf_pages) == 0:
            print(f"  No Akkadian pages found")
            continue
        
        print(f"  {len(pdf_pages)} Akkadian pages")
        
        # Get texts linked to this PDF
        linked_texts = links_df[links_df['pdf_name'] == pdf_name]['oare_id'].unique()
        
        if dry_run:
            print(f"  Would process {len(pdf_pages)} pages for {len(linked_texts)} linked texts")
            continue
        
        # Extract from each page
        for _, page_row in pdf_pages.iterrows():
            extractions = extract_translations_from_page(
                page_row['page_text'],
                model
            )
            
            for ext in extractions:
                all_extractions.append({
                    'pdf_name': pdf_name,
                    'page': page_row['page'],
                    'text_id': ext.get('text_id'),
                    'translation': ext.get('translation'),
                    'language': ext.get('language'),
                    'confidence': ext.get('confidence'),
                    'linked_oare_ids': ','.join(linked_texts)
                })
            
            # Rate limiting
            time.sleep(0.5)
        
        print(f"  Extracted {len([e for e in all_extractions if e['pdf_name'] == pdf_name])} translations")
    
    return pd.DataFrame(all_extractions)


def main():
    parser = argparse.ArgumentParser(description='Extract translations from publications using LLM')
    parser.add_argument('--sample', type=int, help='Process only N PDFs for testing')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without calling LLM')
    parser.add_argument('--api-key', type=str, help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--output', type=str, default='data/processed/extracted_translations.csv',
                        help='Output file path')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    links_path = base_dir / 'data' / 'processed' / 'publication_links.csv'
    pubs_path = base_dir / 'data' / 'publications.csv'
    output_path = base_dir / args.output
    
    # Check for links file
    if not links_path.exists():
        print(f"Error: Run publication_matcher.py first to create {links_path}")
        return
    
    print("Loading data...")
    links_df = pd.read_csv(links_path)
    pubs_df = pd.read_csv(pubs_path)
    
    print(f"Publication links: {len(links_df)}")
    print(f"Unique PDFs with links: {links_df['pdf_name'].nunique()}")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        # Just show statistics
        for pdf in links_df['pdf_name'].unique()[:args.sample or 5]:
            akkadian_pages = len(pubs_df[(pubs_df['pdf_name'] == pdf) & (pubs_df['has_akkadian'] == True)])
            linked = links_df[links_df['pdf_name'] == pdf]['oare_id'].nunique()
            print(f"  {pdf[:50]}: {akkadian_pages} Akkadian pages, {linked} linked texts")
        return
    
    # Setup Gemini
    if not setup_gemini(args.api_key):
        return
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini API configured")
    
    # Process
    extractions_df = process_matches(
        links_df, pubs_df, model,
        sample_size=args.sample,
        dry_run=args.dry_run
    )
    
    if len(extractions_df) > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        extractions_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(extractions_df)} extractions to {output_path}")
        
        # Statistics
        print("\n=== EXTRACTION STATISTICS ===")
        print(f"Total extractions: {len(extractions_df)}")
        print(f"By language:")
        for lang, count in extractions_df['language'].value_counts().items():
            print(f"  {lang}: {count}")
        print(f"High confidence: {len(extractions_df[extractions_df['confidence'] == 'high'])}")
    else:
        print("No extractions found")


if __name__ == '__main__':
    main()
