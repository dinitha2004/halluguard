from models.generator import generate_with_hidden_states

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