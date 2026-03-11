# Phase 2 Plan

## Task 1: Training unit definition

### Main supervised training unit
One token = one training example.

### Token label rule
- 0 = not hallucinated
- 1 = hallucinated

### Why token-level is used
Token-level is used because the research focuses on:
- fine-grained hallucination detection
- exact wrong segment detection
- not just full paragraph classification

### Span-level role
Span-level data is not the main training unit.
It is used for:
- aggregation of consecutive hallucinated tokens
- span-level evaluation
- highlighted segment reporting

### Overall response-level role
Overall response label is not the main training unit.
It is used for:
- final response-level reporting
- overall hallucination score
- UI presentation

### Final hierarchy
1. Token level = main training level
2. Span level = support evaluation level
3. Overall response level = final reporting level