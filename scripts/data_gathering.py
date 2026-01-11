"""
Data Gathering Pipeline for Deep Past Challenge
================================================
This script gathers and consolidates training data from multiple sources:
1. Sentence-level alignment from Sentences_Oare_FirstWord_LinNum.csv
2. Document-level translations from train.csv
3. Translation extraction from publications.csv (OCR text)
4. AICC translation URLs from published_texts.csv
"""

import pandas as pd
import numpy as np
import re
import json
import os
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

# =============================================================================
# 1. SENTENCE-LEVEL ALIGNMENT
# =============================================================================

def create_sentence_pairs():
    """
    Create sentence-level training pairs by matching sentence translations
    with the corresponding portions of document-level transliterations.
    """
    print_section("1. CREATING SENTENCE-LEVEL TRAINING PAIRS")
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    sent_df = pd.read_csv(DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv")
    
    print(f"Training documents: {len(train_df)}")
    print(f"Sentence alignments: {len(sent_df)}")
    
    # Create lookup from text_uuid to oare_id
    # We need to match based on display_name or text_uuid
    
    # Group sentences by text_uuid
    text_sentences = defaultdict(list)
    for _, row in sent_df.iterrows():
        text_sentences[row['text_uuid']].append({
            'sentence_uuid': row['sentence_uuid'],
            'translation': row['translation'],
            'first_word': row['first_word_spelling'],
            'line_number': row['line_number'],
            'sentence_obj': row['sentence_obj_in_text']
        })
    
    print(f"Unique texts in sentence data: {len(text_sentences)}")
    
    # Create a mapping from display names to training data
    # Extract text names from training oare_ids and display_names
    
    sentence_pairs = []
    matched_texts = 0
    
    # For each training document, try to find matching sentences
    for _, train_row in train_df.iterrows():
        oare_id = train_row['oare_id']
        full_translit = train_row['transliteration']
        full_translation = train_row['translation']
        
        # Check if this text_uuid exists in sentence data
        if oare_id in text_sentences:
            matched_texts += 1
            sentences = text_sentences[oare_id]
            
            # Sort sentences by position in text
            sentences = sorted(sentences, key=lambda x: x['sentence_obj'])
            
            # Try to segment the transliteration based on first words
            for sent in sentences:
                if sent['translation'] and sent['first_word']:
                    # Find the first word in the transliteration
                    first_word = sent['first_word']
                    
                    sentence_pairs.append({
                        'source': 'alignment',
                        'text_id': oare_id,
                        'transliteration': None,  # Will be segmented
                        'translation': sent['translation'],
                        'first_word': first_word,
                        'line_number': sent['line_number'],
                        'full_transliteration': full_translit,
                        'full_translation': full_translation
                    })
    
    print(f"Matched training texts: {matched_texts}")
    print(f"Sentence pairs with translations: {len(sentence_pairs)}")
    
    # Now try to extract transliterations for each sentence
    successful_pairs = []
    
    for pair in sentence_pairs:
        if pair['first_word'] and pair['full_transliteration']:
            # Try to find the sentence in the full transliteration
            translit = pair['full_transliteration']
            first_word = pair['first_word']
            
            # Search for the first word
            pattern = re.escape(first_word)
            matches = list(re.finditer(pattern, translit))
            
            if matches:
                # Take the first match for now
                start_pos = matches[0].start()
                
                # Find the end of the sentence (next sentence start or end of document)
                # This is a simplified approach
                remaining = translit[start_pos:]
                
                # Split on common sentence boundaries
                # Akkadian sentences often end with certain patterns
                
                pair['transliteration_segment'] = remaining[:200] if len(remaining) > 200 else remaining
                successful_pairs.append(pair)
    
    print(f"Successfully extracted segments: {len(successful_pairs)}")
    
    return sentence_pairs, successful_pairs

# =============================================================================
# 2. EXTRACT TRANSLATIONS FROM SENTENCE ALIGNMENT DIRECTLY
# =============================================================================

def extract_direct_sentence_translations():
    """
    Extract sentence translations directly from the alignment file.
    These can be used even without perfect transliteration alignment.
    """
    print_section("2. EXTRACTING DIRECT SENTENCE TRANSLATIONS")
    
    sent_df = pd.read_csv(DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv")
    
    # Get all unique sentences with translations
    translations = []
    for _, row in sent_df.iterrows():
        if pd.notna(row['translation']) and row['translation'].strip():
            translations.append({
                'text_uuid': row['text_uuid'],
                'sentence_uuid': row['sentence_uuid'],
                'translation': row['translation'].strip(),
                'first_word_spelling': row['first_word_spelling'] if pd.notna(row['first_word_spelling']) else '',
                'first_word_transcription': row['first_word_transcription'] if pd.notna(row['first_word_transcription']) else '',
                'line_number': row['line_number']
            })
    
    print(f"Total sentence translations: {len(translations)}")
    print(f"Unique texts: {len(set(t['text_uuid'] for t in translations))}")
    
    # Save translations
    trans_df = pd.DataFrame(translations)
    trans_df.to_csv(OUTPUT_DIR / "sentence_translations.csv", index=False)
    print(f"Saved to: {OUTPUT_DIR / 'sentence_translations.csv'}")
    
    return translations

# =============================================================================
# 3. EXTRACT TRANSLATIONS FROM PUBLICATIONS (OCR)
# =============================================================================

def extract_from_publications():
    """
    Extract potential translation pairs from OCR'd publications.
    Look for patterns indicating translations.
    """
    print_section("3. EXTRACTING TRANSLATIONS FROM PUBLICATIONS")
    
    # Load publications
    pub_df = pd.read_csv(DATA_DIR / "publications.csv", low_memory=False)
    print(f"Total publication pages: {len(pub_df)}")
    
    # Filter pages with Akkadian
    akk_pages = pub_df[pub_df['has_akkadian'] == True].copy()
    print(f"Pages with Akkadian: {len(akk_pages)}")
    
    # Patterns that might indicate translations
    translation_patterns = [
        # English translation markers
        r'[Tt]ranslation[:\s]',
        r'"([^"]{10,500})"',  # Quoted text that might be translation
        r"'([^']{10,500})'",  # Single quoted text
        r'[Ss]ays?[:\s]+"([^"]+)"',
        r'[Ss]peak[s]?[:\s]+"([^"]+)"',
        r'[Mm]essage[:\s]+"([^"]+)"',
    ]
    
    extracted_translations = []
    
    for _, row in akk_pages.iterrows():
        page_text = str(row['page_text'])
        pdf_name = row['pdf_name']
        page = row['page']
        
        # Look for translation patterns
        for pattern in translation_patterns:
            matches = re.findall(pattern, page_text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if len(match) > 20 and len(match) < 1000:
                    # Check if it looks like English (not Akkadian)
                    english_words = ['the', 'and', 'to', 'of', 'he', 'she', 'you', 'silver', 'minas']
                    if any(w in match.lower() for w in english_words):
                        extracted_translations.append({
                            'source': 'publication',
                            'pdf_name': pdf_name,
                            'page': page,
                            'translation': match.strip(),
                            'context': page_text[max(0, page_text.find(match)-100):page_text.find(match)+len(match)+100]
                        })
    
    print(f"Extracted potential translations: {len(extracted_translations)}")
    
    # Save extracted translations
    if extracted_translations:
        extract_df = pd.DataFrame(extracted_translations)
        extract_df.to_csv(OUTPUT_DIR / "publication_translations.csv", index=False)
        print(f"Saved to: {OUTPUT_DIR / 'publication_translations.csv'}")
    
    return extracted_translations

# =============================================================================
# 4. GET AICC TRANSLATION URLS
# =============================================================================

def get_aicc_urls():
    """
    Get all AICC translation URLs from published_texts.csv.
    These can potentially be scraped for additional translations.
    """
    print_section("4. COLLECTING AICC TRANSLATION URLS")
    
    pub_texts = pd.read_csv(DATA_DIR / "published_texts.csv")
    
    # Get texts with AICC translations
    has_aicc = pub_texts[pub_texts['AICC_translation'].notna()].copy()
    print(f"Texts with AICC translation URLs: {len(has_aicc)}")
    
    # Save URLs for potential scraping
    aicc_data = has_aicc[['oare_id', 'cdli_id', 'label', 'AICC_translation', 'transliteration']].copy()
    aicc_data.to_csv(OUTPUT_DIR / "aicc_urls.csv", index=False)
    print(f"Saved AICC URLs to: {OUTPUT_DIR / 'aicc_urls.csv'}")
    
    # Count unique URL patterns
    url_patterns = aicc_data['AICC_translation'].str.extract(r'(https?://[^/]+)', expand=False)
    print(f"URL domains: {url_patterns.value_counts().to_dict()}")
    
    return aicc_data

# =============================================================================
# 5. MATCH SENTENCE TRANSLATIONS WITH TRAIN DATA
# =============================================================================

def match_sentences_with_train():
    """
    Match sentence translations with training documents to create
    aligned sentence-level training pairs.
    """
    print_section("5. MATCHING SENTENCES WITH TRAINING DATA")
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    sent_df = pd.read_csv(DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv")
    
    # Group sentences by text_uuid
    text_sentences = sent_df.groupby('text_uuid').apply(
        lambda x: x.sort_values('sentence_obj_in_text').to_dict('records')
    ).to_dict()
    
    # Create aligned pairs
    aligned_pairs = []
    
    for _, train_row in train_df.iterrows():
        oare_id = train_row['oare_id']
        
        if oare_id in text_sentences:
            sentences = text_sentences[oare_id]
            full_translit = train_row['transliteration']
            
            # Try to segment the transliteration
            translit_words = full_translit.split()
            
            for i, sent in enumerate(sentences):
                translation = sent.get('translation', '')
                first_word = sent.get('first_word_spelling', '')
                
                if translation and isinstance(translation, str) and len(translation) > 5:
                    # Try to find the transliteration segment
                    if first_word and first_word in full_translit:
                        # Find start position
                        start_idx = full_translit.find(first_word)
                        
                        # Find end position (next sentence start or end)
                        if i + 1 < len(sentences):
                            next_first = sentences[i+1].get('first_word_spelling', '')
                            if next_first and next_first in full_translit[start_idx+1:]:
                                end_idx = full_translit.find(next_first, start_idx+1)
                            else:
                                end_idx = len(full_translit)
                        else:
                            end_idx = len(full_translit)
                        
                        segment = full_translit[start_idx:end_idx].strip()
                        
                        if segment and len(segment) > 10:
                            aligned_pairs.append({
                                'oare_id': oare_id,
                                'transliteration': segment,
                                'translation': translation,
                                'source': 'aligned',
                                'line_number': sent.get('line_number', ''),
                                'sentence_uuid': sent.get('sentence_uuid', '')
                            })
    
    print(f"Created aligned pairs: {len(aligned_pairs)}")
    
    if aligned_pairs:
        aligned_df = pd.DataFrame(aligned_pairs)
        aligned_df.to_csv(OUTPUT_DIR / "aligned_sentence_pairs.csv", index=False)
        print(f"Saved to: {OUTPUT_DIR / 'aligned_sentence_pairs.csv'}")
    
    return aligned_pairs

# =============================================================================
# 6. CREATE CONSOLIDATED TRAINING SET
# =============================================================================

def create_consolidated_dataset():
    """
    Consolidate all data sources into a unified training set.
    """
    print_section("6. CREATING CONSOLIDATED TRAINING SET")
    
    # Load original training data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    train_df['source'] = 'original_train'
    train_df['level'] = 'document'
    
    all_data = []
    
    # Add original document-level training data
    for _, row in train_df.iterrows():
        all_data.append({
            'transliteration': row['transliteration'],
            'translation': row['translation'],
            'source': 'original_train',
            'level': 'document',
            'oare_id': row['oare_id']
        })
    
    print(f"Original training documents: {len(train_df)}")
    
    # Load aligned sentence pairs if available
    aligned_path = OUTPUT_DIR / "aligned_sentence_pairs.csv"
    if aligned_path.exists():
        aligned_df = pd.read_csv(aligned_path)
        for _, row in aligned_df.iterrows():
            all_data.append({
                'transliteration': row['transliteration'],
                'translation': row['translation'],
                'source': 'aligned_sentences',
                'level': 'sentence',
                'oare_id': row.get('oare_id', '')
            })
        print(f"Aligned sentence pairs: {len(aligned_df)}")
    
    # Create consolidated dataframe
    consolidated_df = pd.DataFrame(all_data)
    
    # Remove duplicates based on translation
    consolidated_df = consolidated_df.drop_duplicates(subset=['translation'], keep='first')
    
    # Save
    consolidated_df.to_csv(OUTPUT_DIR / "consolidated_training_data.csv", index=False)
    print(f"\nTotal consolidated training examples: {len(consolidated_df)}")
    print(f"  - Document level: {len(consolidated_df[consolidated_df['level'] == 'document'])}")
    print(f"  - Sentence level: {len(consolidated_df[consolidated_df['level'] == 'sentence'])}")
    print(f"Saved to: {OUTPUT_DIR / 'consolidated_training_data.csv'}")
    
    return consolidated_df

# =============================================================================
# 7. CREATE SUMMARY STATISTICS
# =============================================================================

def create_summary():
    """Create summary of all gathered data."""
    print_section("7. DATA GATHERING SUMMARY")
    
    summary = {
        "original_train": len(pd.read_csv(DATA_DIR / "train.csv")),
        "sentence_alignments": len(pd.read_csv(DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv")),
        "published_texts": len(pd.read_csv(DATA_DIR / "published_texts.csv")),
        "publication_pages": len(pd.read_csv(DATA_DIR / "publications.csv", low_memory=False)),
        "lexicon_entries": len(pd.read_csv(DATA_DIR / "OA_Lexicon_eBL.csv")),
        "dictionary_entries": len(pd.read_csv(DATA_DIR / "eBL_Dictionary.csv")),
    }
    
    # Check processed files
    if (OUTPUT_DIR / "aligned_sentence_pairs.csv").exists():
        summary["aligned_sentence_pairs"] = len(pd.read_csv(OUTPUT_DIR / "aligned_sentence_pairs.csv"))
    if (OUTPUT_DIR / "consolidated_training_data.csv").exists():
        summary["consolidated_training_data"] = len(pd.read_csv(OUTPUT_DIR / "consolidated_training_data.csv"))
    if (OUTPUT_DIR / "aicc_urls.csv").exists():
        summary["aicc_urls"] = len(pd.read_csv(OUTPUT_DIR / "aicc_urls.csv"))
    
    print("\nData Summary:")
    print("-" * 50)
    for key, value in summary.items():
        print(f"  {key}: {value:,}")
    
    # Save summary
    with open(OUTPUT_DIR / "data_gathering_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {OUTPUT_DIR / 'data_gathering_summary.json'}")
    
    return summary

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" DEEP PAST CHALLENGE - DATA GATHERING PIPELINE")
    print("="*70)
    
    # Run all steps
    sentence_pairs, successful_pairs = create_sentence_pairs()
    direct_translations = extract_direct_sentence_translations()
    pub_translations = extract_from_publications()
    aicc_data = get_aicc_urls()
    aligned_pairs = match_sentences_with_train()
    consolidated = create_consolidated_dataset()
    summary = create_summary()
    
    print("\n" + "="*70)
    print(" DATA GATHERING COMPLETE!")
    print("="*70)
    print(f"\nProcessed files saved to: {OUTPUT_DIR}")
