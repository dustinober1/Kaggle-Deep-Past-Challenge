#!/usr/bin/env python3
"""
AICC Scraper - Phase 1.2 Alternative Extraction

Scrapes AI translations from aicuneiform.com using their JSON API.
No LLM or browser automation needed!

Input: published_texts.csv (AICC_translation URLs)
Output: data/processed/aicc_translations.csv
"""

import argparse
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import pandas as pd


def parse_translation(html: str) -> Tuple[str, str]:
    """Parse HTML to extract Akkadian and translation using regex."""
    if not html:
        return ('', '')
    
    # Unescape HTML entities
    html = html.replace('\\n', '\n').replace('&quot;', '"').replace('&amp;', '&')
    
    akkadian_lines = []
    translation_lines = []
    
    # Split HTML by language section markers
    # Each section starts with <div class='lang-XXX
    parts = re.split(r"(<div class='lang-[^']+)", html)
    
    current_lang = None
    for i, part in enumerate(parts):
        if part.startswith("<div class='lang-"):
            # Extract language type
            if 'lang-akk' in part:
                current_lang = 'akk'
            elif 'lang-ml_en' in part:
                current_lang = 'en'
            else:
                current_lang = None
        elif current_lang and i > 0:
            # Extract span content from this section
            lines = re.findall(r"<span class='line[^']*'>([^<]+)</span>", part)
            if current_lang == 'akk':
                akkadian_lines.extend(lines)
            elif current_lang == 'en':
                translation_lines.extend(lines)
    
    return (
        ' '.join(akkadian_lines),
        ' '.join(translation_lines)
    )


def extract_p_number(url: str) -> Optional[str]:
    """Extract P-number from AICC URL."""
    # URL format: https://aicuneiform.com/search?q=P361099
    match = re.search(r'[Pp](\d+)', url)
    if match:
        return f"P{match.group(1)}"
    return None


def fetch_publication_json(p_number: str, cache: Dict[str, dict]) -> Optional[dict]:
    """Fetch publication JSON from AICC API."""
    # API pattern: /p/pXXX.json where XXX is first 4 chars of P-number (lowercase)
    prefix = p_number[:4].lower()
    json_url = f"https://aicuneiform.com/p/{prefix}.json"
    
    if json_url in cache:
        return cache[json_url]
    
    try:
        req = Request(json_url, headers={'User-Agent': 'Mozilla/5.0 Deep Past Challenge Research'})
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            cache[json_url] = data
            return data
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"  Error fetching {json_url}: {e}")
        return None


def scrape_aicc_translations(
    texts_df: pd.DataFrame,
    sample_size: Optional[int] = None,
    delay: float = 0.5
) -> pd.DataFrame:
    """Scrape translations for texts with AICC URLs."""
    
    # Filter to texts with AICC URLs
    has_url = texts_df[texts_df['AICC_translation'].notna()].copy()
    print(f"Texts with AICC URLs: {len(has_url)}")
    
    if sample_size:
        has_url = has_url.head(sample_size)
        print(f"Sampling first {sample_size}")
    
    # Group by P-number prefix to minimize API calls
    results = []
    json_cache = {}
    
    for i, (_, row) in enumerate(has_url.iterrows()):
        if i > 0 and i % 100 == 0:
            print(f"Progress: {i}/{len(has_url)}")
        
        url = row['AICC_translation']
        p_number = extract_p_number(url)
        
        if not p_number:
            continue
        
        # Fetch JSON (may be cached)
        pub_data = fetch_publication_json(p_number, json_cache)
        
        if not pub_data or p_number not in pub_data:
            continue
        
        # Parse HTML
        html = pub_data[p_number].get('html', '')
        akkadian, translation = parse_translation(html)
        
        if akkadian or translation:
            results.append({
                'oare_id': row['oare_id'],
                'p_number': p_number,
                'aicc_akkadian': akkadian,
                'aicc_translation': translation,
                'existing_transliteration': row.get('transliteration', ''),
            })
        
        # Rate limiting (minimal since we cache by prefix)
        if p_number[:4] not in [r.get('p_number', '')[:4] for r in results[:-1]]:
            time.sleep(delay)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Scrape AICC translations')
    parser.add_argument('--sample', type=int, help='Process only N texts for testing')
    parser.add_argument('--output', type=str, default='data/processed/aicc_translations.csv')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between API calls')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    texts_path = base_dir / 'data' / 'published_texts.csv'
    train_path = base_dir / 'data' / 'train.csv'
    output_path = base_dir / args.output
    
    print("Loading data...")
    texts_df = pd.read_csv(texts_path)
    train_df = pd.read_csv(train_path)
    
    # Exclude texts already in train.csv
    train_ids = set(train_df['oare_id'])
    texts_df = texts_df[~texts_df['oare_id'].isin(train_ids)]
    print(f"Texts not in train.csv: {len(texts_df)}")
    
    print("\nScraping AICC translations...")
    results_df = scrape_aicc_translations(
        texts_df,
        sample_size=args.sample,
        delay=args.delay
    )
    
    print(f"\n=== RESULTS ===")
    print(f"Scraped translations: {len(results_df)}")
    print(f"With Akkadian text: {(results_df['aicc_akkadian'].str.len() > 0).sum()}")
    print(f"With translation: {(results_df['aicc_translation'].str.len() > 0).sum()}")
    
    if len(results_df) > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
        
        # Show sample
        print("\n=== SAMPLE OUTPUT ===")
        for _, row in results_df.head(2).iterrows():
            print(f"\noare_id: {row['oare_id']}")
            print(f"P-number: {row['p_number']}")
            print(f"AICC Akkadian: {row['aicc_akkadian'][:100]}...")
            print(f"AICC Translation: {row['aicc_translation'][:100]}...")


if __name__ == '__main__':
    main()
