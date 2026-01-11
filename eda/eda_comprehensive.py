"""
Comprehensive EDA for Deep Past Challenge: Akkadian to English Translation
===========================================================================
This script performs exploratory data analysis on all competition datasets.
"""

import pandas as pd
import numpy as np
import os
import json
import re
from collections import Counter
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

DATA_DIR = Path(__file__).parent.parent / "data"

def separator(title):
    """Print a section separator"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def analyze_text_column(series, name="text"):
    """Analyze a text column and return statistics"""
    # Remove NaN values
    texts = series.dropna()
    
    # Length statistics (characters)
    char_lengths = texts.str.len()
    
    # Word count (split by whitespace)
    word_counts = texts.str.split().str.len()
    
    # Token count (split by whitespace and hyphens for Akkadian)
    token_counts = texts.str.replace('-', ' ').str.split().str.len()
    
    stats = {
        "count": len(texts),
        "char_length_mean": char_lengths.mean(),
        "char_length_std": char_lengths.std(),
        "char_length_min": char_lengths.min(),
        "char_length_max": char_lengths.max(),
        "char_length_median": char_lengths.median(),
        "word_count_mean": word_counts.mean(),
        "word_count_std": word_counts.std(),
        "word_count_min": word_counts.min(),
        "word_count_max": word_counts.max(),
        "word_count_median": word_counts.median(),
        "token_count_mean": token_counts.mean(),
        "token_count_median": token_counts.median(),
    }
    return stats, char_lengths, word_counts

def extract_special_patterns(texts):
    """Extract special patterns from Akkadian transliterations"""
    patterns = {
        "determinatives": r'\{[^}]+\}',  # {d}, {ki}, etc.
        "logograms_caps": r'\b[A-ZŠṢṬḪ][A-ZŠṢṬḪ0-9.]+\b',  # ALL CAPS words (Sumerian logograms)
        "proper_nouns": r'\b[A-ZŠṢṬḪ][a-zšṣṭḫàáèéìíùú][a-zšṣṭḫàáèéìíùú0-9-]*\b',  # Capitalized words
        "gaps": r'<gap>|<big_gap>|\[x\]|…|\[\.\.\.\]',
        "numbers": r'\b\d+\b',
        "broken_signs": r'\[[^\]]+\]|˹[^˺]+˺',
        "uncertain_readings": r'\?',
        "word_dividers": r'[:/]',
    }
    
    results = {}
    combined_text = ' '.join(texts.dropna().astype(str))
    
    for name, pattern in patterns.items():
        matches = re.findall(pattern, combined_text)
        results[name] = {
            "count": len(matches),
            "unique": len(set(matches)),
            "examples": list(set(matches))[:10]
        }
    
    return results

def analyze_vocabulary(texts, is_akkadian=False):
    """Analyze vocabulary in text"""
    all_words = []
    for text in texts.dropna():
        if is_akkadian:
            # For Akkadian, split on whitespace and keep hyphens
            words = str(text).split()
        else:
            # For English, standard tokenization
            words = re.findall(r'\b\w+\b', str(text).lower())
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    
    return {
        "total_tokens": len(all_words),
        "unique_tokens": len(word_freq),
        "type_token_ratio": len(word_freq) / len(all_words) if all_words else 0,
        "most_common_20": word_freq.most_common(20),
        "hapax_legomena": sum(1 for count in word_freq.values() if count == 1),
    }

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    
    separator("DEEP PAST CHALLENGE - COMPREHENSIVE EDA")
    print("Competition: Translate Akkadian to English")
    print("Task: Neural Machine Translation of Old Assyrian cuneiform texts")
    print()
    
    # =========================================================================
    # 1. TRAIN DATA ANALYSIS
    # =========================================================================
    separator("1. TRAINING DATA ANALYSIS (train.csv)")
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"Shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    print(f"\nData types:\n{train_df.dtypes}")
    print(f"\nMissing values:\n{train_df.isnull().sum()}")
    print(f"\nFirst few rows:")
    print(train_df.head(3))
    
    print("\n--- Transliteration (Akkadian) Statistics ---")
    trans_stats, trans_chars, trans_words = analyze_text_column(train_df['transliteration'], "transliteration")
    for key, value in trans_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n--- Translation (English) Statistics ---")
    eng_stats, eng_chars, eng_words = analyze_text_column(train_df['translation'], "translation")
    for key, value in eng_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n--- Text Length Comparison ---")
    print(f"  Avg Akkadian chars per doc: {trans_stats['char_length_mean']:.1f}")
    print(f"  Avg English chars per doc: {eng_stats['char_length_mean']:.1f}")
    print(f"  Ratio (Eng/Akk): {eng_stats['char_length_mean']/trans_stats['char_length_mean']:.2f}")
    print(f"  Avg Akkadian words per doc: {trans_stats['word_count_mean']:.1f}")
    print(f"  Avg English words per doc: {eng_stats['word_count_mean']:.1f}")
    
    print("\n--- Special Patterns in Transliterations ---")
    patterns = extract_special_patterns(train_df['transliteration'])
    for name, data in patterns.items():
        print(f"  {name}: {data['count']} occurrences, {data['unique']} unique")
        if data['examples']:
            print(f"    Examples: {data['examples'][:5]}")
    
    print("\n--- Akkadian Vocabulary ---")
    akk_vocab = analyze_vocabulary(train_df['transliteration'], is_akkadian=True)
    print(f"  Total tokens: {akk_vocab['total_tokens']:,}")
    print(f"  Unique tokens: {akk_vocab['unique_tokens']:,}")
    print(f"  Type-token ratio: {akk_vocab['type_token_ratio']:.4f}")
    print(f"  Hapax legomena: {akk_vocab['hapax_legomena']:,} ({100*akk_vocab['hapax_legomena']/akk_vocab['unique_tokens']:.1f}% of vocabulary)")
    print(f"  Most common tokens: {[w for w,c in akk_vocab['most_common_20'][:10]]}")
    
    print("\n--- English Vocabulary ---")
    eng_vocab = analyze_vocabulary(train_df['translation'], is_akkadian=False)
    print(f"  Total tokens: {eng_vocab['total_tokens']:,}")
    print(f"  Unique tokens: {eng_vocab['unique_tokens']:,}")
    print(f"  Type-token ratio: {eng_vocab['type_token_ratio']:.4f}")
    print(f"  Hapax legomena: {eng_vocab['hapax_legomena']:,} ({100*eng_vocab['hapax_legomena']/eng_vocab['unique_tokens']:.1f}% of vocabulary)")
    print(f"  Most common tokens: {[w for w,c in eng_vocab['most_common_20'][:10]]}")
    
    # =========================================================================
    # 2. TEST DATA ANALYSIS
    # =========================================================================
    separator("2. TEST DATA ANALYSIS (test.csv)")
    
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
    print(f"\nData types:\n{test_df.dtypes}")
    print(f"\nMissing values:\n{test_df.isnull().sum()}")
    print(f"\nFirst few rows:")
    print(test_df.head(3))
    
    print(f"\n--- Test Set Structure ---")
    print(f"  Total sentences: {len(test_df)}")
    print(f"  Unique documents (text_id): {test_df['text_id'].nunique()}")
    print(f"  Sentences per document: {len(test_df)/test_df['text_id'].nunique():.1f} avg")
    
    print("\n--- Sentence Transliteration Statistics ---")
    test_stats, test_chars, test_words = analyze_text_column(test_df['transliteration'], "test_transliteration")
    for key, value in test_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n--- Line Numbers Analysis ---")
    print(f"  line_start unique values: {test_df['line_start'].nunique()}")
    print(f"  line_start examples: {test_df['line_start'].head(10).tolist()}")
    
    # =========================================================================
    # 3. PUBLISHED TEXTS ANALYSIS  
    # =========================================================================
    separator("3. PUBLISHED TEXTS ANALYSIS (published_texts.csv)")
    
    pub_df = pd.read_csv(DATA_DIR / "published_texts.csv")
    print(f"Shape: {pub_df.shape}")
    print(f"Columns: {list(pub_df.columns)}")
    print(f"\nData types:\n{pub_df.dtypes}")
    print(f"\nMissing values (showing columns with missing data):")
    missing = pub_df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False).head(20))
    
    print(f"\n--- Overlap Analysis ---")
    train_ids = set(train_df['oare_id'].values)
    pub_ids = set(pub_df['oare_id'].values)
    print(f"  Training text IDs: {len(train_ids)}")
    print(f"  Published text IDs: {len(pub_ids)}")
    print(f"  Overlap (train texts in published): {len(train_ids & pub_ids)}")
    print(f"  Published texts NOT in training: {len(pub_ids - train_ids)}")
    
    print(f"\n--- Genre Distribution ---")
    if 'genre_label' in pub_df.columns:
        genre_counts = pub_df['genre_label'].value_counts()
        print(genre_counts.head(15))
    
    print(f"\n--- Has AICC Translation? ---")
    if 'AICC_translation' in pub_df.columns:
        has_aicc = pub_df['AICC_translation'].notna().sum()
        print(f"  Texts with AICC translation URL: {has_aicc}")
    
    # =========================================================================
    # 4. LEXICON ANALYSIS
    # =========================================================================
    separator("4. LEXICON ANALYSIS (OA_Lexicon_eBL.csv)")
    
    lex_df = pd.read_csv(DATA_DIR / "OA_Lexicon_eBL.csv")
    print(f"Shape: {lex_df.shape}")
    print(f"Columns: {list(lex_df.columns)}")
    print(f"\nData types:\n{lex_df.dtypes}")
    print(f"\nFirst few rows:")
    print(lex_df.head(5))
    
    print(f"\n--- Word Type Distribution ---")
    if 'type' in lex_df.columns:
        type_counts = lex_df['type'].value_counts()
        print(type_counts)
    
    print(f"\n--- Unique Forms and Lexemes ---")
    if 'form' in lex_df.columns:
        print(f"  Unique forms: {lex_df['form'].nunique()}")
    if 'lexeme' in lex_df.columns:
        print(f"  Unique lexemes: {lex_df['lexeme'].nunique()}")
    if 'norm' in lex_df.columns:
        print(f"  Unique normalized forms: {lex_df['norm'].nunique()}")
    
    # =========================================================================
    # 5. DICTIONARY ANALYSIS
    # =========================================================================
    separator("5. DICTIONARY ANALYSIS (eBL_Dictionary.csv)")
    
    dict_df = pd.read_csv(DATA_DIR / "eBL_Dictionary.csv")
    print(f"Shape: {dict_df.shape}")
    print(f"Columns: {list(dict_df.columns)}")
    print(f"\nData types:\n{dict_df.dtypes}")
    print(f"\nFirst few rows:")
    print(dict_df.head(5))
    
    # =========================================================================
    # 6. BIBLIOGRAPHY ANALYSIS
    # =========================================================================
    separator("6. BIBLIOGRAPHY ANALYSIS (bibliography.csv)")
    
    bib_df = pd.read_csv(DATA_DIR / "bibliography.csv")
    print(f"Shape: {bib_df.shape}")
    print(f"Columns: {list(bib_df.columns)}")
    print(f"\nFirst few rows:")
    print(bib_df.head(5))
    
    print(f"\n--- Publication Year Distribution ---")
    if 'year' in bib_df.columns:
        year_counts = bib_df['year'].value_counts().sort_index()
        print(f"  Year range: {bib_df['year'].min()} - {bib_df['year'].max()}")
        print(f"  Most publications in years: {bib_df['year'].value_counts().head(5)}")
    
    # =========================================================================
    # 7. SENTENCE ALIGNMENT DATA
    # =========================================================================
    separator("7. SENTENCE ALIGNMENT DATA (Sentences_Oare_FirstWord_LinNum.csv)")
    
    sent_df = pd.read_csv(DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv")
    print(f"Shape: {sent_df.shape}")
    print(f"Columns: {list(sent_df.columns)}")
    print(f"\nFirst few rows:")
    print(sent_df.head(10))
    
    print(f"\n--- Sentence Counts per Document ---")
    if 'oare_id' in sent_df.columns:
        sent_per_doc = sent_df.groupby('oare_id').size()
        print(f"  Total documents: {sent_per_doc.count()}")
        print(f"  Avg sentences per doc: {sent_per_doc.mean():.1f}")
        print(f"  Min sentences: {sent_per_doc.min()}")
        print(f"  Max sentences: {sent_per_doc.max()}")
    
    # =========================================================================
    # 8. SAMPLE SUBMISSION
    # =========================================================================
    separator("8. SAMPLE SUBMISSION ANALYSIS")
    
    sample_df = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print(f"Shape: {sample_df.shape}")
    print(f"Columns: {list(sample_df.columns)}")
    print(f"\nFirst few rows:")
    print(sample_df.head(5))
    
    # =========================================================================
    # 9. KEY INSIGHTS SUMMARY
    # =========================================================================
    separator("9. KEY INSIGHTS AND CHALLENGES")
    
    print("""
DATASET SUMMARY:
================
1. Training Data: ~1,500 document-level translations
   - Full documents with Akkadian transliteration + English translation
   - Avg document length: ~{:.0f} words Akkadian, ~{:.0f} words English

2. Test Data: ~{} sentence-level translations needed
   - {} unique documents split into sentences
   - Need to predict English translations for each sentence

3. Supplemental Data:
   - ~8,000 published transliterations (metadata, no translations)
   - ~880 scholarly publications (OCR text, potential translations)
   - Comprehensive lexicon and dictionary

KEY CHALLENGES:
===============
1. LOW RESOURCE: Only ~1,500 training examples
2. DOMAIN SHIFT: Training=document-level, Test=sentence-level
3. COMPLEX MORPHOLOGY: Akkadian has rich morphology (single word = multiple English words)
4. SPECIAL NOTATIONS: Determinatives {{}}, logograms (CAPS), gaps, broken signs
5. PROPER NOUNS: Names need special handling
6. DATA QUALITY: OCR artifacts, formatting inconsistencies

POTENTIAL STRATEGIES:
====================
1. Extract additional translations from publications.csv
2. Use lexicon for vocabulary augmentation
3. Sentence-level alignment of training data
4. Handle special tokens (<gap>, determinatives, etc.)
5. Fine-tune multilingual models (mBART, NLLB, etc.)
6. Use pretrained language models with domain adaptation
""".format(
        trans_stats['word_count_mean'],
        eng_stats['word_count_mean'],
        len(test_df),
        test_df['text_id'].nunique()
    ))
    
    # =========================================================================
    # 10. SAVE DETAILED STATISTICS
    # =========================================================================
    separator("10. SAVING DETAILED STATISTICS")
    
    stats_output = {
        "train": {
            "num_documents": len(train_df),
            "akkadian_stats": trans_stats,
            "english_stats": eng_stats,
            "akkadian_vocab": {k: v for k, v in akk_vocab.items() if k != 'most_common_20'},
            "english_vocab": {k: v for k, v in eng_vocab.items() if k != 'most_common_20'},
            "special_patterns": {k: {"count": v["count"], "unique": v["unique"]} for k, v in patterns.items()},
        },
        "test": {
            "num_sentences": len(test_df),
            "num_documents": test_df['text_id'].nunique(),
            "transliteration_stats": test_stats,
        },
        "published_texts": {
            "total_texts": len(pub_df),
            "overlap_with_train": len(train_ids & pub_ids),
            "texts_without_training": len(pub_ids - train_ids),
        },
        "lexicon": {
            "total_entries": len(lex_df),
        },
        "dictionary": {
            "total_entries": len(dict_df),
        },
    }
    
    output_path = DATA_DIR / "processed" / "eda_statistics.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    stats_clean = json.loads(json.dumps(stats_output, default=convert_numpy))
    
    with open(output_path, 'w') as f:
        json.dump(stats_clean, f, indent=2)
    
    print(f"Statistics saved to: {output_path}")
    
    print("\n" + "="*80)
    print(" EDA COMPLETE")
    print("="*80)
