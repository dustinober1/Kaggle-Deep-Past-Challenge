# Deep Past Challenge - Exploratory Data Analysis Report

## Competition Overview

**Goal:** Build neural machine translation models that convert transliterated Akkadian (Old Assyrian) into English.

**Evaluation Metric:** Geometric Mean of BLEU and chrF++ scores (micro-averaged across the corpus)

**Prize Pool:** $50,000 (1st: $15K, 2nd: $10K, 3rd: $8K, 4th: $7K, 5th: $5K, 6th: $5K)

**Constraints:**
- CPU Notebook ≤ 9 hours
- GPU Notebook ≤ 9 hours  
- No internet access during inference
- Freely available external data and pretrained models allowed

---

## 1. Training Data (`train.csv`)

### Overview
| Metric | Value |
|--------|-------|
| Total Documents | **1,561** |
| Columns | `oare_id`, `transliteration`, `translation` |
| Missing Values | None |

### Text Statistics

#### Akkadian Transliterations
| Statistic | Value |
|-----------|-------|
| Avg Characters | 426.5 |
| Avg Words (whitespace-split) | 57.5 |
| Avg Tokens (including hyphens) | 134.3 |
| Min Words | 3 |
| Max Words | 187 |
| Median Words | 49 |

#### English Translations  
| Statistic | Value |
|-----------|-------|
| Avg Characters | 499.7 |
| Avg Words | 90.5 |
| Min Words | 1 |
| Max Words | 744 |
| Median Words | 68 |

### Length Ratio
- English translations are ~17% longer than Akkadian in characters
- English has ~1.57x more words than Akkadian tokens
- This reflects Akkadian's morphological richness (1 word = multiple English words)

### Vocabulary Analysis

#### Akkadian Vocabulary
| Metric | Value |
|--------|-------|
| Total Tokens | 89,807 |
| Unique Tokens | 11,761 |
| Type-Token Ratio | 0.131 |
| Hapax Legomena | 7,033 (59.8% of vocab) |

**Most Common Akkadian Tokens:**
`a-na`, `ša`, `KÙ.BABBAR`, `x`, `ma-na`, `DUMU`, `GÍN`, `ù`, `…`, `i-na`

#### English Vocabulary
| Metric | Value |
|--------|-------|
| Total Tokens | 149,462 |
| Unique Tokens | 4,511 |
| Type-Token Ratio | 0.030 |
| Hapax Legomena | 1,785 (39.6% of vocab) |

**Most Common English Words:**
`of`, `the`, `and`, `to`, `silver`, `aššur`, `you`, `i`, `for`, `son`

### Special Patterns Found

| Pattern Type | Occurrences | Unique | Examples |
|-------------|-------------|--------|----------|
| **Logograms (CAPS)** | 12,461 | 168 | `DUMU`, `KÙ.BABBAR`, `GÍN`, `TÚG` |
| **Gaps/Breaks** | 1,897 | 2 | `[...]`, `…` |
| **Numbers** | 8,386 | 96 | `1`, `9`, `31`, `81` |
| **Broken Signs** | 210 | 1 | `[...]` |

### Key Observations

1. **High Hapax Rate:** 59.8% of Akkadian vocabulary appears only once → severe data sparsity
2. **Sumerian Logograms:** Many texts contain Sumerian words in ALL CAPS (e.g., `KÙ.BABBAR` = silver)
3. **Domain-Specific:** Texts are commercial (silver, textiles, contracts)
4. **Missing Determinatives:** Expected `{d}`, `{ki}` patterns not found - may need preprocessing

---

## 2. Test Data (`test.csv`)

### Overview
| Metric | Value |
|--------|-------|
| Sample Size (provided) | 4 sentences |
| Actual Test Size | ~4,000 sentences |
| Unique Documents | ~400 |
| Columns | `id`, `text_id`, `line_start`, `line_end`, `transliteration` |

### Critical Insight: Domain Shift
- **Training:** Document-level (full tablets, ~58 words avg)
- **Test:** Sentence-level (partial tablets, ~21 words avg)
- This requires sentence-level alignment of training data!

### Sentence Statistics (Sample)
| Statistic | Value |
|-----------|-------|
| Avg Characters | 168.8 |
| Avg Words | 21.3 |
| Avg Tokens | 54.3 |

---

## 3. Published Texts (`published_texts.csv`)

### Overview
| Metric | Value |
|--------|-------|
| Total Texts | **7,953** |
| Columns | 19 (metadata + transliterations) |

### Data Availability
| Field | Available | Missing |
|-------|-----------|---------|
| Transliteration | 7,953 | 0 |
| AICC Translation URLs | 7,702 | 251 |
| Genre Label | 7,943 | 10 |
| CDLI ID | 7,702 | 251 |
| eBL ID | 1,472 | 6,481 |

### Overlap with Training
- **Training texts in published:** 1,561 (100%)
- **Published texts NOT in training:** 6,388
- **Potential additional data:** 6,388 untranslated texts

### Genre Distribution
| Genre | Count | % |
|-------|-------|---|
| unknown | 4,046 | 50.9% |
| letter | 2,261 | 28.4% |
| debt note | 527 | 6.6% |
| note | 218 | 2.7% |
| agreement (contract) | 128 | 1.6% |
| legal writing(s) | 94 | 1.2% |
| testimony | 83 | 1.0% |
| memo | 80 | 1.0% |
| other | 516 | 6.6% |

---

## 4. Lexicon (`OA_Lexicon_eBL.csv`)

### Overview
| Metric | Value |
|--------|-------|
| Total Entries | **39,332** |
| Unique Forms | 35,048 |
| Unique Lexemes | 6,353 |
| Unique Normalized Forms | 19,662 |

### Word Type Distribution
| Type | Count | % |
|------|-------|---|
| word | 25,574 | 65.0% |
| PN (Person Name) | 13,424 | 34.1% |
| GN (Geographic Name) | 334 | 0.8% |

### Utility
- Maps surface forms to dictionary lemmas
- Can be used for:
  - Vocabulary expansion
  - Named entity recognition
  - Morphological analysis
  - Creating word embeddings

---

## 5. Dictionary (`eBL_Dictionary.csv`)

### Overview
| Metric | Value |
|--------|-------|
| Total Entries | **19,215** |
| Columns | `word`, `definition`, `derived_from` |

### Sample Entries
| Word | Definition |
|------|------------|
| -ak(k)um I | "like" comparative suffix on subst.s, adj.s, and prons. |
| -āku I | "my" suffix for 1 sg. genitive (stative, pret.) |
| -ī I | (adv. endings) |
| -iš I | "to; like" term.-adv. suffix on subst.s and adj.s |

---

## 6. Bibliography (`bibliography.csv`)

### Overview
| Metric | Value |
|--------|-------|
| Total Publications | **878** |
| Year Range | 1882 - 2025 |

### Publication Years (Top 5)
| Year | Publications |
|------|--------------|
| 2008 | 47 |
| 1977 | 37 |
| 2003 | 36 |
| 2004 | 35 |
| 2001 | 33 |

---

## 7. Sentence Alignment Data (`Sentences_Oare_FirstWord_LinNum.csv`)

### Overview
| Metric | Value |
|--------|-------|
| Total Entries | **9,782** |
| Columns | 12 (including `sentence_uuid`, `translation`, `line_number`) |

### Utility
- Provides sentence-level translations for training data
- Maps sentences to specific line numbers on tablets
- **Critical for creating sentence-level training pairs**

---

## 8. Resources (`resources.csv`)

Contains bibliographic references to computational approaches to Akkadian and cuneiform, including:
- Machine translation research
- OCR systems
- Neural network approaches
- Historical linguistic tools

---

## Key Challenges & Strategies

### Challenge 1: Extremely Low Resource
**Problem:** Only 1,561 training documents  
**Strategies:**
- Extract additional translations from `publications.csv` (OCR)
- Use the sentence alignment data to create sentence-level pairs
- Leverage AICC translations (poor quality but available)
- Data augmentation techniques

### Challenge 2: Domain Shift (Document → Sentence)
**Problem:** Training is document-level, test is sentence-level  
**Strategies:**
- Use `Sentences_Oare_FirstWord_LinNum.csv` for alignment
- Split training documents into sentences
- Train on sentence pairs, not full documents

### Challenge 3: Complex Morphology
**Problem:** Akkadian words encode what takes multiple English words  
**Strategies:**
- BPE/Subword tokenization (e.g., SentencePiece)
- Morphological analysis preprocessing
- Use lexicon for vocabulary expansion

### Challenge 4: Special Notations
**Problem:** Determinatives, logograms, gaps, broken signs  
**Strategies:**
- Standardize notation (convert to `<gap>`, `<big_gap>`)
- Keep logograms as-is (they're meaningful)
- Special tokens for damaged portions
- Consider removing scribal annotations (`!`, `?`, `/`)

### Challenge 5: Named Entities
**Problem:** 13,424 person names + 334 place names  
**Strategies:**
- Copy mechanism in decoder
- Named entity recognition preprocessing
- Keep proper nouns capitalized

### Challenge 6: Evaluation Metric
**Problem:** Geometric mean of BLEU and chrF++ rewards both precision and character-level accuracy  
**Strategies:**
- Optimize for both metrics during training
- Character-level models may help with chrF++
- Ensemble approaches

---

## Recommended Next Steps

1. **Data Preparation**
   - [ ] Create sentence-level training pairs using alignment data
   - [ ] Preprocess transliterations (standardize gaps, notations)
   - [ ] Extract additional translations from publications
   - [ ] Build vocabulary with subword tokenization

2. **Model Development**
   - [ ] Baseline: Fine-tune mBART/NLLB-200 on Akkadian→English
   - [ ] Try character-level models for chrF++ boost
   - [ ] Experiment with copy mechanisms for names
   - [ ] Consider back-translation for data augmentation

3. **Training Strategy**
   - [ ] Start with pretrained multilingual models
   - [ ] Mixed document/sentence training
   - [ ] Curriculum learning (easy→hard)
   - [ ] Ensemble diverse models

4. **Evaluation**
   - [ ] Implement exact competition metric locally
   - [ ] Cross-validation on training data
   - [ ] Error analysis on named entities, numbers, gaps

---

## Data File Summary

| File | Rows | Key Purpose |
|------|------|-------------|
| `train.csv` | 1,561 | Document-level training pairs |
| `test.csv` | ~4,000 | Sentence-level inference (actual) |
| `published_texts.csv` | 7,953 | Additional unlabeled transliterations |
| `OA_Lexicon_eBL.csv` | 39,332 | Word forms → lexemes mapping |
| `eBL_Dictionary.csv` | 19,215 | Akkadian definitions |
| `bibliography.csv` | 878 | Publication metadata |
| `Sentences_Oare_FirstWord_LinNum.csv` | 9,782 | Sentence alignment helper |
| `sample_submission.csv` | 4 | Submission format |

---

*Generated: Deep Past Challenge EDA*
