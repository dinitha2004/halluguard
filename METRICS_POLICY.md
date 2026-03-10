# HalluGuard Metrics Policy

## 1. Token-level metrics
The official token-level metrics are:
- Precision
- Recall
- F1-score

These measure how accurately the system identifies hallucinated tokens.

### Primary token-level metric
- Token F1-score

## 2. Span-level metrics
The official span-level metrics are:
- Precision
- Recall
- F1-score

These measure how accurately the system identifies hallucinated spans.

### Primary span-level metric
- Span F1-score

## 3. Overall response-level metrics
The official response-level metrics are:
- Accuracy
- Macro F1-score

These measure how accurately the system classifies the full response risk.

### Primary response-level metric
- Overall Macro F1-score

## 4. Reporting rule
Final reported performance must be measured only on the held-out test set.

## 5. Validation rule
The validation set may be used for:
- threshold tuning
- model selection
- score combination tuning

The validation set must not be used as the final reported result.

## 6. Core metric summary
The three main project metrics are:
- Token F1-score
- Span F1-score
- Overall Macro F1-score