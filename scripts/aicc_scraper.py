"""
AICC Translation Scraper
========================
Scrapes translations from aicuneiform.com URLs.
Note: Uses respectful rate limiting and caching.
"""

import pandas as pd
import requests
import time
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "processed"
CACHE_DIR = OUTPUT_DIR / "aicc_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Rate limiting
REQUESTS_PER_SECOND = 2
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND

def get_cache_path(url):
    """Generate cache file path for URL."""
    # Create safe filename from URL
    safe_name = re.sub(r'[^\w]', '_', url.split('=')[-1] if '=' in url else url.split('/')[-1])
    return CACHE_DIR / f"{safe_name}.json"

def fetch_aicc_translation(url, session=None):
    """
    Fetch translation from an AICC URL.
    Returns dict with translation or None if failed.
    """
    cache_path = get_cache_path(url)
    
    # Check cache first
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    
    try:
        if session is None:
            session = requests.Session()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Academic Research - Akkadian Translation Project)'
        }
        
        response = session.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find translation content
            # AICC structure varies, try multiple approaches
            result = {
                'url': url,
                'status': 'success',
                'translation': None,
                'transliteration': None,
                'metadata': {}
            }
            
            # Look for translation divs/sections
            for div in soup.find_all(['div', 'section', 'p']):
                text = div.get_text(strip=True)
                
                # Look for translation markers
                if 'Translation' in text or 'translation' in text:
                    # Get the content after "Translation"
                    trans_match = re.search(r'[Tt]ranslation[:\s]*(.+)', text, re.DOTALL)
                    if trans_match:
                        result['translation'] = trans_match.group(1)[:2000]
                
                # Look for transliteration
                if 'Transliteration' in text or any(c in text for c in ['ša', 'um-ma', 'a-na']):
                    if not result['transliteration']:
                        result['transliteration'] = text[:2000]
            
            # Try finding specific elements
            for elem in soup.find_all(class_=re.compile(r'translation|trans', re.I)):
                if not result['translation']:
                    result['translation'] = elem.get_text(strip=True)[:2000]
            
            # Cache result
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            return result
        else:
            return {'url': url, 'status': 'error', 'error': f'HTTP {response.status_code}'}
            
    except Exception as e:
        return {'url': url, 'status': 'error', 'error': str(e)}

def scrape_aicc_translations(max_urls=None, batch_size=100):
    """
    Scrape translations from AICC URLs.
    Uses rate limiting and caching.
    """
    print("="*70)
    print("AICC Translation Scraper")
    print("="*70)
    
    # Load URLs
    aicc_df = pd.read_csv(OUTPUT_DIR / "aicc_urls.csv")
    print(f"Total AICC URLs: {len(aicc_df)}")
    
    if max_urls:
        aicc_df = aicc_df.head(max_urls)
        print(f"Processing first {max_urls} URLs")
    
    # Check what's already cached
    cached_count = sum(1 for url in aicc_df['AICC_translation'] if get_cache_path(url).exists())
    print(f"Already cached: {cached_count}")
    
    results = []
    session = requests.Session()
    
    urls_to_process = [
        (row['oare_id'], row['AICC_translation'], row['transliteration'])
        for _, row in aicc_df.iterrows()
        if pd.notna(row['AICC_translation'])
    ]
    
    print(f"URLs to process: {len(urls_to_process)}")
    print(f"Estimated time: {len(urls_to_process) * REQUEST_DELAY / 60:.1f} minutes")
    
    for i, (oare_id, url, translit) in enumerate(urls_to_process):
        if i > 0 and i % 50 == 0:
            print(f"Progress: {i}/{len(urls_to_process)} ({100*i/len(urls_to_process):.1f}%)")
        
        result = fetch_aicc_translation(url, session)
        
        if result and result.get('status') == 'success':
            results.append({
                'oare_id': oare_id,
                'url': url,
                'transliteration': translit,
                'aicc_translation': result.get('translation', ''),
                'aicc_transliteration': result.get('transliteration', ''),
                'status': 'success'
            })
        else:
            results.append({
                'oare_id': oare_id,
                'url': url,
                'transliteration': translit,
                'status': 'error',
                'error': result.get('error', 'Unknown error') if result else 'No response'
            })
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "aicc_translations.csv", index=False)
    
    success_count = len(results_df[results_df['status'] == 'success'])
    with_trans = len(results_df[results_df['aicc_translation'].notna() & (results_df['aicc_translation'] != '')])
    
    print(f"\nResults:")
    print(f"  Successful requests: {success_count}")
    print(f"  With translations: {with_trans}")
    print(f"  Saved to: {OUTPUT_DIR / 'aicc_translations.csv'}")
    
    return results_df

def check_aicc_sample():
    """
    Check a sample AICC URL to understand the structure.
    """
    print("="*70)
    print("Checking AICC URL Sample")
    print("="*70)
    
    aicc_df = pd.read_csv(OUTPUT_DIR / "aicc_urls.csv")
    sample_url = aicc_df['AICC_translation'].iloc[0]
    
    print(f"Sample URL: {sample_url}")
    
    try:
        response = requests.get(sample_url, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Show page structure
            print(f"\nPage title: {soup.title.string if soup.title else 'No title'}")
            
            # Find main content
            main_divs = soup.find_all('div', class_=True)[:10]
            print(f"\nMain div classes found:")
            for div in main_divs:
                classes = div.get('class', [])
                text_preview = div.get_text(strip=True)[:100]
                if text_preview:
                    print(f"  {classes}: {text_preview}...")
            
            # Save raw HTML for inspection
            with open(OUTPUT_DIR / "aicc_sample.html", 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"\nSaved HTML to: {OUTPUT_DIR / 'aicc_sample.html'}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check_aicc_sample()
    elif len(sys.argv) > 1 and sys.argv[1] == '--scrape':
        max_urls = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        scrape_aicc_translations(max_urls=max_urls)
    else:
        print("Usage:")
        print("  python aicc_scraper.py --check    # Check sample URL structure")
        print("  python aicc_scraper.py --scrape N # Scrape N URLs")
