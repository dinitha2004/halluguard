from models.generator import generate_with_hidden_states
from models.aggregator import (
    build_risky_spans,
    compute_overall_hallucination_score,
    get_overall_label,
)

prompt = "What is the capital of Japan?"

response, token_data = generate_with_hidden_states(prompt, max_new_tokens=30)

print("\nFINAL RESPONSE:\n")
print(response)

print("\nTOKEN DATA PREVIEW:\n")
for item in token_data[:10]:
    print("Step:", item["step"])
    print("Token:", repr(item["token_text"]))
    print("Token ID:", item["token_id"])
    print("Hidden state length:", len(item["hidden_state"]))
    print("SEP score:", item["sep_score"])
    print("HalluShift score:", item["hallushift_score"])
    print("Final risk score:", item["final_risk_score"])
    print("Risk label:", item["risk_label"])
    print("-" * 40)

spans = build_risky_spans(token_data, min_label="MEDIUM")
overall_score = compute_overall_hallucination_score(token_data, spans)
overall_label = get_overall_label(overall_score)

print("\nSPAN AGGREGATOR PREVIEW:\n")
if spans:
    for i, span in enumerate(spans, start=1):
        print("Span:", i)
        print("Text:", repr(span["span_text"]))
        print("Steps:", f"{span['start_step']} to {span['end_step']}")
        print("Token count:", span["token_count"])
        print("Average risk:", span["avg_risk"])
        print("Max risk:", span["max_risk"])
        print("Span label:", span["span_label"])
        print("=" * 50)
else:
    print("No candidate spans detected.")

print("\nOVERALL HALLUCINATION PREVIEW:\n")
print("Overall score:", overall_score)
print("Overall label:", overall_label)