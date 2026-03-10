# HalluGuard Train Validation Test Split Policy

## 1. Split ratio
The dataset will be divided into:
- 70% training
- 15% validation
- 15% test

## 2. Split unit
Splitting is performed at the full sample level, not at the token level.

Each sample includes:
- question
- reference answer
- generated response
- token sequence
- token labels
- span labels

A single sample must belong to exactly one split.

## 3. Purpose of each split

### Training set
Used to train the hallucination probe model.

### Validation set
Used for:
- threshold tuning
- model selection
- score weight tuning
- deciding LOW, MEDIUM, and HIGH risk boundaries

### Test set
Used only for final evaluation and reporting.

## 4. Leakage prevention rules
1. Tokens from the same response must not appear in different splits.
2. Duplicate or near-duplicate samples should not be placed across splits.
3. Thresholds must not be tuned on the test set.
4. Final reported performance must come from the held-out test set.

## 5. Reproducibility
A fixed random seed must be used for dataset splitting.

## 6. Output files
The final split files will be stored in:
- data/splits/train.jsonl
- data/splits/val.jsonl
- data/splits/test.jsonl