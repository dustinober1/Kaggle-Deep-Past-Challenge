# Kaggle - Deep Past Challenge

## Project Status
- **Date**: 2026-01-11
- **Current Task**: Phase 1.2 Publication Translation Extraction (scripts ready)

## Phase 1.2 Results
- Publication links: 7,502 matches to 56 PDFs
- Unique texts matched: 3,770
- New texts (not in train.csv): 2,587
- Pipeline scripts ready for API execution

## Sentence Alignment Results (Phase 1.1)
- Overlapping documents: 253 (train ∩ sentences)
- Aligned sentence pairs: 1,213
- Output: `data/processed/sentence_aligned_train.csv`

## Dataset Summary
| Dataset | Rows | Purpose |
|---------|------|---------|
| train.csv | 1,561 | Labeled translations |
| published_texts.csv | 7,953 | All transliterations |
| publications.csv | 216,602 | OCR from 952 PDFs |
| publication_links.csv | 7,502 | Text-to-PDF links |

## Key Metrics
- Training samples: 1,561 (document-level)
- Potential new samples: 2,587 (via publication extraction)
- Word expansion ratio: 1.48× (Akkadian → English)

## Scripts
- `scripts/sentence_alignment.py` - Phase 1.1
- `scripts/publication_matcher.py` - Link texts to PDFs ✓
- `scripts/translation_extractor.py` - LLM extraction (requires API key)
- `scripts/translation_pipeline.py` - DE/FR → EN
- `scripts/align_and_filter.py` - Quality filtering

## Changelog
- Initialized git repository and `.venv`
- Created EDA in `eda/` with `report.md`
- **Phase 1.1**: Sentence alignment (1,213 pairs)
- **Phase 1.2**: Publication matcher (7,502 links created)
- **Phase 1.2**: Translation extraction pipeline ready

