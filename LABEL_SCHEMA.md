# HalluGuard Label Schema

## 1. Target hallucination type
This project focuses on factual hallucination in LLM generated answers.
A hallucination is defined as a generated factual claim or detail that is incorrect, fabricated, or unsupported by the reference truth.

## 2. Token label definition
Token labels are binary:

- 0 = not hallucinated
- 1 = hallucinated

A token is labeled hallucinated only if it belongs to the exact incorrect factual segment in the response.

## 3. What is not hallucination
The following are not labeled as hallucination by themselves:
1. Grammar mistakes
2. Spelling issues that do not change factual meaning
3. Style differences
4. Hedging language such as "I think" or "maybe"
5. Short but correct answers
6. Missing detail, unless the response introduces a wrong fact

## 4. Span label definition
A span is a contiguous group of risky tokens detected by the model.

Span labels:
- LOW = little or no hallucination evidence
- MEDIUM = suspicious factual segment that should be reviewed
- HIGH = strong evidence that the span contains hallucinated content

## 5. Overall label definition
The overall label summarizes hallucination risk for the full response.

Overall labels:
- LOW = response is mostly trustworthy
- MEDIUM = response contains one or more suspicious factual segments
- HIGH = response is likely unreliable due to strong hallucination evidence

## 6. Initial thresholds
These are provisional thresholds for the prototype stage and may be tuned later using validation data.

Span label:
- LOW = below 0.40
- MEDIUM = 0.40 to 0.69
- HIGH = 0.70 and above

Overall label:
- LOW = below 0.40
- MEDIUM = 0.40 to 0.69
- HIGH = 0.70 and above

## 7. Annotation rules
1. Label only the exact incorrect factual token or phrase.
2. Do not label surrounding correct context unless it is part of the wrong factual segment.
3. If a whole entity name is wrong, label the full entity name.
4. If a number or date is wrong, label only the wrong number or date expression.
5. If multiple separated wrong segments exist, label them separately.
6. If the answer is fully correct, all token labels must be 0.

## 8. Examples

### Example 1
Question: What is the capital of Japan?
Response: The capital of Japan is Tokyo.
Token labels: [0, 0, 0, 0, 0, 0, 0]

### Example 2
Question: What is the capital of Japan?
Response: The capital of Japan is Osaka.
Token labels: [0, 0, 0, 0, 0, 1, 0]

### Example 3
Question: Who wrote Hamlet?
Response: Hamlet was written by Charles Dickens.
Token labels: [0, 0, 0, 0, 1, 1, 0]