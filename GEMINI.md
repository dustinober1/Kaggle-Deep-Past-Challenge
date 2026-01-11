# Kaggle - Deep Past Challenge

## Project Status
- **Date**: 2026-01-11
- **Current Task**: Exploratory Data Analysis Complete

## Dataset Summary
| Dataset | Rows | Purpose |
|---------|------|---------|
| train.csv | 1,561 | Labeled translations |
| published_texts.csv | 7,953 | All transliterations |
| publications.csv | 216,602 | OCR from 952 PDFs |
| OA_Lexicon_eBL.csv | 39,332 | Word forms |
| eBL_Dictionary.csv | 19,215 | Definitions |
| Sentences.csv | 9,782 | Sentence alignments |

## Key Metrics
- Training samples: 1,561 (document-level)
- Untranslated texts: 6,388 (potential for extraction)
- Word expansion ratio: 1.48× (Akkadian → English)
- Lexicon coverage: 18.2% of train vocabulary

## Changelog
- Initialized git repository
- Created `.venv` virtual environment
- Added `.gitignore`
- Formatted competition rules and documentation in `rules/`
- Created `eda/` directory with 6 analysis scripts
- Generated visualizations in `eda/figures/`
- Completed comprehensive `report.md` with EDA findings
- Created `todo.md` with competition roadmap
