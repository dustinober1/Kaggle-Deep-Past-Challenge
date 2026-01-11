# Kaggle - Deep Past Challenge

## Project Status
- **Date**: 2026-01-11
- **Current Task**: Phase 7 Evaluation & Error Analysis Completed ✅

## Merged Training Dataset
- **Total samples: 6,316** (4× increase!)
  - Original train.csv: 1,561
  - AICC scraped: 4,755
- Output: `data/processed/augmented_train.csv`

## Phase 7 Results
- **Evaluation Metrics**: BLEU, chrF++, geometric mean via SacreBLEU
- **Error Analysis**: Categorized by named entities, numbers, gaps, word order
- **Validation Strategy**: Cross-validation, document vs sentence comparison
- **Per-Example Scoring**: Detailed breakdown with patterns

## Phase 6 Results
- **Training Infrastructure**: TensorBoard logging, checkpointing
- **Multi-Level Training**:
  - Document-level: 1,296 train / 149 val
  - Sentence-level: 1,024 train / 119 val
  - Curriculum & mixed modes implemented
- **Optimization**: Gradient accumulation, Adam/AdamW/Adafactor, label smoothing, dropout

## Phase 1.3 Results
- **Lexicon Enhancement**: Built 3 lookup resources
  - `word_dictionary.json`: 13,718 definitions
  - `form_to_lexeme.json`: 22,053 form mappings
  - `named_entities.json`: 13,422 PNs, 334 GNs
- **Coverage**: 10.7% of training tokens matched to lexicon forms (baseline)

## Phase 1.2 Results
- **AICC Scraper**: 77% success rate (4,755 / 6,141 texts)
  - Uses JSON API at `/p/pXXX.json`
  - No LLM or browser automation needed
  - Output: `data/processed/aicc_translations.csv`
- Publication links: 7,502 matches to 56 PDFs
- Unique texts matched: 3,770

## Sentence Alignment Results (Phase 1.1)
- Overlapping documents: 253 (train ∩ sentences)
- Aligned sentence pairs: 1,213
- Output: `data/processed/sentence_aligned_train.csv`

## Dataset Summary
| Dataset | Rows | Purpose |
|---------|------|---------|
| augmented_train.csv | 6,316 | **Merged training set** |
| train.csv | 1,561 | Original labeled translations |
| aicc_translations.csv | 4,755 | AICC scraped translations |
| published_texts.csv | 7,953 | All transliterations |
| publications.csv | 216,602 | OCR from 952 PDFs |

## Key Metrics
- Training samples: 6,316 (merged)
- Avg transliteration length: 469 chars
- Avg translation length: 548 chars
- Word expansion ratio: 1.48× (Akkadian → English)

## Scripts
- `scripts/aicc_scraper.py` - AICC JSON API scraper ✓
- `scripts/merge_aicc_data.py` - Merge with train.csv ✓
- `scripts/sentence_alignment.py` - Phase 1.1 ✓
- `scripts/publication_matcher.py` - Link texts to PDFs ✓
- `scripts/translation_extractor.py` - LLM extraction
- `scripts/translation_pipeline.py` - DE/FR → EN
- `scripts/align_and_filter.py` - Quality filtering
- `scripts/dataset_loader.py` - Akkadian dataset loader ✓ (multi-level)
- `scripts/finetune_pretrained.py` - mBART/NLLB/M2M fine-tuning ✓
- `scripts/custom_transformer.py` - Custom model architecture (with ALiBi) ✓
- `scripts/train_custom.py` - Custom model training loop ✓ (TensorBoard, grad accum)
- `scripts/compare_models.py` - Model comparison suite ✓
- `scripts/multi_level_trainer.py` - Multi-level training modes ✓
- `scripts/optimization_experiments.py` - Hyperparameter sweeps ✓
- `scripts/evaluation_metrics.py` - BLEU, chrF++, geometric mean scoring ✓
- `scripts/error_analyzer.py` - Error categorization by type ✓
- `scripts/evaluate.py` - Unified evaluation driver ✓

## Changelog
- Initialized git repository and `.venv`
- Created EDA in `eda/` with `report.md`
- **Phase 1.1**: Sentence alignment (1,213 pairs)
- **Phase 1.2**: Publication matcher (7,502 links created)
- **Phase 1.2**: AICC scraper completed (4,755 translations)
- **Phase 1.2**: Merged training data (6,316 samples)
- **Phase 4**: Tokenizer training (SentencePiece)
- **Phase 5**: Baseline Model Development
  - Implemented unified fine-tuning script for mBART/NLLB/M2M with logging
  - Implemented Custom Transformer with ALiBi, Noam Scheduler, Checkpointing
  - Verified training loop execution on MPS
  - Created comparison script
- **Phase 6**: Training Pipeline
  - Added TensorBoard logging to train_custom.py and finetune_pretrained.py
  - Added gradient accumulation, optimizer selection, label smoothing, dropout
  - Created multi-level training: document/sentence/curriculum/mixed modes
  - Created optimization_experiments.py for hyperparameter sweeps
- **Phase 7**: Evaluation & Error Analysis
  - Implemented BLEU, chrF++, geometric mean with SacreBLEU
  - Created error analyzer for named entities, numbers, gaps, word order
  - Added cross-validation and document vs sentence comparison
  - Created unified evaluate.py driver script

