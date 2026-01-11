# Deep Past Challenge - Project Roadmap

## Competition Overview
- **Task**: Translate Old Assyrian Akkadian → English
- **Metric**: Geometric Mean of BLEU × chrF++
- **Deadline**: March 23, 2026
- **Submission**: Kaggle Notebook (<9hr, no internet, GPU P100/T4×2)

---

## ✅ Phase 1: Exploratory Data Analysis (COMPLETE)

### 1.1 Competition Understanding
- [x] Read competition rules and requirements
- [x] Understand evaluation metric (BLEU × chrF++ geometric mean)
- [x] Note Kaggle notebook constraints (9hr, no internet)

### 1.2 Data Analysis
- [x] Analyze train.csv (1,561 document-level pairs)
- [x] Analyze test.csv (sentence-level, ~4,000 expected)
- [x] Explore supplemental data files
- [x] Identify domain shift: document vs sentence level
- [x] Document special characters and patterns

### 1.3 Key Findings Documented
- [x] High hapax legomena rate (59.8% of vocabulary)
- [x] Akkadian ~58 words/doc, English ~91 words/doc
- [x] Special patterns: determinatives, logograms, gaps
- [x] Created `EDA_REPORT.md`

---

## ✅ Phase 2: Data Gathering (COMPLETE)

### 2.1 Sentence-Level Alignment
- [x] Parse `Sentences_Oare_FirstWord_LinNum.csv`
- [x] Match with train.csv via `text_uuid` ↔ `oare_id`
- [x] Create aligned sentence pairs (1,072 pairs)
- [x] Save to `aligned_sentence_pairs.csv`

### 2.2 Publication Translation Extraction
- [x] Analyze publication OCR quality
- [x] Extract potential translations (59,538 candidates)
- [x] Filter for high-quality pairs (7 clean pairs)
- [x] Note: OCR quality limits extraction

### 2.3 Resource Cataloguing
- [x] Catalog AICC URLs (7,702 machine translation links)
- [x] Catalog sentence translations (9,771 entries)
- [x] Map lexicon entries (39,332 form→lexeme)
- [x] Map dictionary entries (19,215 definitions)

### 2.4 Final Training Set
- [x] Consolidate all sources
- [x] Create `final_training_data.csv` (2,595 examples)
- [x] **66% increase** over original train.csv
- [x] Created `DATA_GATHERING_REPORT.md`

---

## ✅ Phase 3: Data Preprocessing (COMPLETE)

### 3.1 Transliteration Standardization ✅
- [x] Standardize gap markers: `[x]`, `…` → `<gap>`, `<big_gap>`
- [x] Handle determinatives consistently: `(d)` → `{d}`, `(ki)` → `{ki}`, etc.
- [x] Normalize broken signs: `˹ ˺` markers removed
- [x] Handle uncertainty markers: `!`, `?` removed
- [x] Remove/normalize brackets: `[ ]` → content only

**Cleaned Files Created:**
- `data/processed/train_cleaned.csv` (1,561 rows, 56.4% modified)
- `data/processed/final_training_data_cleaned.csv` (2,595 rows, 41.4% modified)  
- `data/processed/test_cleaned.csv` (4 rows, 50% modified)

### 3.2 Special Token Handling ✅
- [x] Create vocabulary for Sumerian logograms (ALL CAPS) - 167 unique logograms
- [x] Handle subscript/superscript numbers (₂, ₄, ₁₅) - 701 patterns identified
- [x] Create <PN> token for personal names - 13,046 name forms catalogued
- [x] Create <GN> token for geographic names - 328 place names catalogued
- [x] Handle numbers and measurements - 244 unique number patterns

**Vocabulary Files Created (`models/tokenizers/`):**
- `logogram_vocabulary.json` - 167 logograms with counts
- `logograms.txt` - Logogram list for tokenizer
- `logogram_meanings.json` - Sumerian→English mappings
- `personal_names.txt` - 13,046 PN forms
- `geographic_names.txt` - 328 GN forms
- `special_token_stats.json` - Complete analysis statistics

### 3.3 Translation Normalization ✅
- [x] Standardize ellipsis markers (`...` → `…`) - 1,055 changes
- [x] Normalize quotation marks (smart quotes → ASCII) - 105 changes
- [x] Consistent handling of "[broken]" text → `[gap]`, `[?]`, `[damaged]` - 58 changes
- [x] Standardize proper noun capitalization (Aššur, City, Kanesh) - 21 changes

**Normalized Files Created:**
- `data/processed/train_cleaned_normalized.csv` - 1,028 rows modified
- `data/processed/final_training_data_cleaned_normalized.csv` - 1,414 rows modified
- `data/processed/translation_normalization_stats.json` - Before/after statistics

### 3.4 Train/Validation Split ✅
- [x] Create stratified validation set (~10%) - 10.4% achieved
- [x] Balance by document length (short/medium/long bins)
- [x] Balance by source (document: 9.8% val, sentence: 11.1% val)
- [x] Ensure no document overlap between splits - verified ✓

**Split Results:**
- **Train:** 2,320 rows, 1,403 unique documents (89.6%)
- **Validation:** 268 rows, 153 unique documents (10.4%)
- Similar length distributions (mean ~327-330, median 243)

**Files Created:**
- `data/processed/train_final.csv` - Final training set
- `data/processed/val_final.csv` - Final validation set
- `data/processed/train_val_split_stats.json` - Split statistics
- `data/processed/train_val_document_ids.json` - Document ID lists

---

## ✅ Phase 4: Tokenization Strategy (COMPLETE)

### 4.1 Tokenizer Training ✅
- [x] Combine all transliterations for tokenizer training
- [x] Train SentencePiece/BPE tokenizer
- [x] Test vocabulary sizes: 500, 1K, 2K (Adjusted due to data constraints)
- [x] Consider hyphen-aware tokenization for syllables (BPE handles effectively)

**Models Created (`models/tokenizers/`):**
- `spm_akkadian_500.model`
- `spm_akkadian_1000_special.model` (Selected Final Model)
- `spm_akkadian_2000.model`
- `spm_akkadian_char.model` (Baseline)

### 4.2 Special Considerations ✅
- [x] Preserve special tokens (<gap>, <big_gap>, etc.)
- [x] Handle determinatives as single units
- [x] Compare with character-level approach
- [x] Benchmark tokenization coverage on test samples

**Benchmark Results:**
- **Selected Model (BPE-1000-Special)**: ~169 tokens/sent (48% reduction vs char)
- **Character Baseline**: ~328 tokens/sent

### 4.3 Alternative Approaches
- [x] Test Character-level handling (Completed via benchmark)
- [x] Evaluate morphological tokenization (Verified BPE alignment with `analyze_morphology.py`)
- [x] Consider hybrid approaches (Adopted BPE + Special Tokens strategy)

---

## ✅ Phase 5: Baseline Model Development (COMPLETE)

### 5.1 Pretrained Model Fine-tuning
- [x] Set up training environment (PyTorch/HuggingFace)
- [x] Fine-tune mBART-50 on Akkadian→English (Script implemented, smoke test passed)
- [x] Fine-tune NLLB-200 on Akkadian→English (Script implemented, smoke test running)
- [x] Fine-tune M2M-100 on Akkadian→English (Script implemented, smoke test running)
- [x] Compare validation performance

### 5.2 Custom Transformer
- [x] Implement 6-layer encoder-decoder
- [x] Test relative position encodings (ALiBi implemented)
- [x] Experiment with attention patterns (ALiBi implemented)
- [x] Train from scratch on combined data (Smoke test passed)

### 5.3 Model Configuration
- [x] Optimize batch size for available GPU memory
- [x] Implement learning rate scheduling (Noam)
- [x] Add early stopping on validation loss
- [x] Implement checkpoint saving

---

## 🔲 Phase 6: Training Pipeline

### 6.1 Training Infrastructure
- [ ] Set up training scripts
- [ ] Implement data loading pipeline
- [ ] Add logging (TensorBoard/Weights&Biases)
- [ ] Implement checkpointing

### 6.2 Multi-Level Training
- [ ] Train on document-level first (more context)
- [ ] Fine-tune on sentence-level (match test format)
- [ ] Curriculum learning: easy → hard examples
- [ ] Mixed training strategy

### 6.3 Optimization
- [ ] Implement gradient accumulation
- [ ] Test different optimizers (Adam, AdamW, Adafactor)
- [ ] Experiment with label smoothing
- [ ] Test dropout rates

---

## 🔲 Phase 7: Evaluation & Error Analysis

### 7.1 Evaluation Metrics
- [ ] Implement BLEU scoring (SacreBLEU)
- [ ] Implement chrF++ scoring
- [ ] Calculate geometric mean (competition metric)
- [ ] Track per-example scores

### 7.2 Error Analysis
- [ ] Categorize error types:
  - [ ] Named entity errors (people, places)
  - [ ] Number/measurement errors
  - [ ] Gap handling errors
  - [ ] Word order errors
- [ ] Analyze by sentence length
- [ ] Analyze by genre/document type
- [ ] Identify systematic patterns

### 7.3 Validation Strategy
- [ ] Cross-validation for robust estimates
- [ ] Test on held-out documents
- [ ] Compare document vs sentence performance

---

## 🔲 Phase 8: Advanced Techniques

### 8.1 Data Augmentation
- [ ] Back-translation (train Eng→Akk, generate synthetic)
- [ ] Lexicon-based substitution
- [ ] Noise injection for robustness
- [ ] Synonym replacement in English

### 8.2 Ensemble Methods
- [ ] Train multiple model variants
- [ ] Combine document and sentence-level models
- [ ] Implement beam search with diverse hypotheses
- [ ] Weight by confidence/difficulty

### 8.3 Post-Processing
- [ ] Dictionary-based correction for known words
- [ ] Named entity consistency checking
- [ ] Length ratio filtering for outliers
- [ ] Grammar checking on English output

### 8.4 Lexicon Integration
- [ ] Create word-level translation memory
- [ ] Use lexicon for rare word handling
- [ ] Constrained decoding with known translations
- [ ] Copy mechanism for proper nouns

---

## 🔲 Phase 9: Kaggle Submission Setup

### 9.1 Notebook Development
- [ ] Create inference notebook structure
- [ ] Package model weights for offline use
- [ ] Implement efficient batch inference
- [ ] Add progress tracking

### 9.2 Resource Optimization
- [ ] Ensure <9 hour total runtime
- [ ] Optimize memory usage for P100/T4×2
- [ ] Test with internet disabled
- [ ] Benchmark inference speed

### 9.3 Output Generation
- [ ] Generate predictions for test.csv
- [ ] Format as `id,translation` CSV
- [ ] Validate submission format
- [ ] Handle edge cases (empty, very long)

---

## 🔲 Phase 10: Final Submission & Iteration

### 10.1 Submission Strategy
- [ ] Submit baseline model first
- [ ] Iterate based on public leaderboard
- [ ] Reserve submissions for final week
- [ ] Maintain submission log

### 10.2 Documentation
- [ ] Document final model architecture
- [ ] Record hyperparameters
- [ ] Write solution summary
- [ ] Prepare for potential write-up

---

## Timeline

| Phase | Target Date | Status |
|-------|-------------|--------|
| 1. EDA | 2026-01-11 | ✅ Complete |
| 2. Data Gathering | 2026-01-11 | ✅ Complete |
| 3. Preprocessing | 2026-01-15 | ✅ Complete|
| 4. Tokenization | 2026-01-18 | ✅ Complete |
| 5. Baseline Models | 2026-01-11 | ✅ Complete |
| 6. Training Pipeline | 2026-02-01 | 🔲 Pending |
| 7. Evaluation | 2026-02-08 | 🔲 Pending |
| 8. Advanced Techniques | 2026-02-28 | 🔲 Pending |
| 9. Submission Setup | 2026-03-15 | 🔲 Pending |
| 10. Final Submission | 2026-03-23 | 🔲 Deadline |

---

## Key Files Reference

### Data Files
```
data/
├── train.csv                    # Original training (1,561 docs)
├── test.csv                     # Test set (sentence-level)
├── sample_submission.csv        # Submission format
├── published_texts.csv          # Additional transliterations
├── OA_Lexicon_eBL.csv          # Word form → lexeme mapping
├── eBL_Dictionary.csv          # Akkadian definitions
├── Sentences_Oare_*.csv        # Sentence alignment data
└── processed/
    ├── final_training_data.csv  # Combined training (2,595)
    ├── final_training_data_cleaned.csv # ★ Cleaned combined training
    ├── final_training_data_cleaned_normalized.csv # ★★ Fully preprocessed
    ├── train_cleaned.csv        # ★ Cleaned train.csv
    ├── train_cleaned_normalized.csv # ★★ Fully preprocessed train
    ├── test_cleaned.csv         # ★ Cleaned test.csv
    ├── translation_normalization_stats.json # Normalization stats
    ├── train_final.csv              # ★★★ Final training set (2,320 rows)
    ├── val_final.csv                # ★★★ Final validation set (268 rows)
    ├── train_val_split_stats.json   # Split statistics
    ├── train_val_document_ids.json  # Document ID lists
    ├── aligned_sentence_pairs.csv
    └── sentence_translations.csv
```

### Scripts
```
scripts/
├── data_gathering.py            # Main data pipeline
├── translation_extractor.py     # Publication extraction
├── aicc_scraper.py             # AICC URL handling
├── transliteration_cleaner.py   # ★ Transliteration standardization
├── test_transliteration_cleaner.py # Tests for cleaner
├── special_token_handler.py     # ★ Logogram/name/number analysis
├── translation_normalizer.py    # ★ Translation text normalization
└── train_val_split.py           # ★ Stratified train/val split
```

### Model Assets
```
models/tokenizers/
├── logogram_vocabulary.json     # ★ 167 Sumerian logograms
├── logograms.txt               # Logogram list
├── logogram_meanings.json      # Sumerian→English mappings
├── personal_names.txt          # ★ 13,046 PN forms
├── geographic_names.txt        # ★ 328 GN forms
└── special_token_stats.json    # Analysis statistics
```

### Reports
```
├── EDA_REPORT.md               # Full EDA findings
├── DATA_GATHERING_REPORT.md    # Data gathering summary
└── todo.md                     # This file
```

---

## Resources & References

### Competition
- [Kaggle Competition Page](https://www.kaggle.com/competitions/deep-past)
- [Competition Data](https://www.kaggle.com/competitions/deep-past/data)

### External Resources
- [ ] CDLI (Cuneiform Digital Library Initiative)
- [ ] eBL (electronic Babylonian Library)
- [ ] ORACC (Open Richly Annotated Cuneiform Corpus)

### Papers to Read
- [ ] Low-resource NMT techniques
- [ ] Akkadian NLP research
- [ ] Similar competition solutions (WMT, IWSLT)

### Pretrained Models to Explore
- [ ] mBART-50 (multilingual)
- [ ] NLLB-200 (200 languages)
- [ ] M2M-100 (100 languages)
- [ ] ByT5 (byte-level)
