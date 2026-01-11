# Deep Past Challenge - Next Steps

## Phase 1: Data Augmentation (Priority: High)

### 1.1 Sentence Alignment Integration
- [x] Parse `Sentences_Oare_FirstWord_LinNum.csv` for sentence-level pairs
- [x] Join with train.csv via `text_uuid` ↔ `oare_id`
- [x] Create sentence-aligned training dataset
- [x] Validate alignment quality on sample

### 1.2 Publication Translation Extraction
- [x] Build ID matching pipeline (aliases → pdf_name)
- [x] Extract translation sections using LLM
- [x] Translate German/French translations to English
- [x] Align extracted translations with published_texts.csv
- [x] Quality filter extracted pairs

### 1.3 Lexicon Enhancement
- [x] Build word-level translation memory from eBL_Dictionary
- [x] Create lookup table for common patterns
- [x] Handle personal names (PN) and geographic names (GN) specially

---

## Phase 2: Preprocessing Pipeline

### 2.1 Transliteration Cleaning
- [x] Remove scribal annotations: `!`, `?`, `˹ ˺`
- [x] Standardize gaps: `[x]` → `<gap>`, `…` → `<big_gap>`
- [x] Normalize brackets: `[ ]` → content only
- [x] Handle determinatives: `{d}`, `{ki}` consistently

### 2.2 Tokenization Strategy
- [ ] Train SentencePiece/BPE on all transliterations
- [ ] Consider hyphen-aware tokenization for syllables
- [ ] Test vocab sizes: 8K, 16K, 32K
- [ ] Compare with character-level approach

### 2.3 Translation Normalization
- [ ] Standardize ellipsis and gap markers
- [ ] Handle quoted speech consistently
- [ ] Normalize proper noun capitalization

---

## Phase 3: Baseline Models

### 3.1 Fine-tune Pretrained MT
- [ ] mBART-50 fine-tuning
- [ ] NLLB-200 fine-tuning
- [ ] M2M-100 fine-tuning
- [ ] Compare performance on validation split

### 3.2 Train From Scratch
- [ ] Transformer (6-layer encoder-decoder)
- [ ] Test attention patterns on Akkadian
- [ ] Experiment with relative position encodings

### 3.3 Character-Level Models
- [ ] ByteT5 or ByT5 for morphological handling
- [ ] May help with OOV words

---

## Phase 4: Evaluation & Iteration

### 4.1 Validation Strategy
- [ ] Create held-out validation set (~10%)
- [ ] Implement geometric mean of BLEU × chrF++
- [ ] Use SacreBLEU for consistent scoring
- [ ] Test on sentence vs document level

### 4.2 Error Analysis
- [ ] Categorize error types (names, numbers, gaps)
- [ ] Compare performance by genre
- [ ] Identify systematic translation errors

---

## Phase 5: Advanced Techniques

### 5.1 Ensemble Methods
- [ ] Combine document-level and sentence-level models
- [ ] Weight by translation difficulty
- [ ] Beam search with diverse hypotheses

### 5.2 Post-Processing
- [ ] Dictionary-based correction for known words
- [ ] Name/number consistency checking
- [ ] Length ratio filtering for outliers

### 5.3 Back-Translation
- [ ] Train English → Akkadian model
- [ ] Generate synthetic training pairs
- [ ] Filter by round-trip consistency

---

## Phase 6: Submission

### 6.1 Kaggle Notebook Setup
- [ ] Create inference notebook
- [ ] Ensure <9 hour runtime
- [ ] Disable internet access
- [ ] Package model weights

### 6.2 Final Submission
- [ ] Generate predictions on test set
- [ ] Format as `submission.csv`
- [ ] Submit and iterate

---

## Timeline

| Phase | Target Date | Status |
|-------|-------------|--------|
| EDA | 2026-01-11 | ✅ Complete |
| Data Augmentation | 2026-01-18 | 🔲 Not started |
| Preprocessing | 2026-01-22 | 🟡 In progress |
| Baseline Models | 2026-02-05 | 🔲 Not started |
| Advanced Techniques | 2026-02-28 | 🔲 Not started |
| Final Submission | 2026-03-23 | 🔲 Deadline |

---

## Resources to Explore

- [ ] Review top solutions from similar low-resource MT competitions
- [ ] Read papers on Akkadian NLP: [Hugging Face collections](https://huggingface.co/collections)
- [ ] Check CDLI and eBL websites for additional data
- [ ] Explore pretrained models for Semitic languages
