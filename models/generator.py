import torch
from models.load_model import load_model
from models.probe import compute_sep_score
from models.hallushift import compute_hallushift_score

tokenizer, model, device = load_model()


def generate_with_hidden_states(prompt: str, max_new_tokens: int = 80):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly and briefly."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids))

    generated_ids = input_ids.clone()
    past_key_values = None

    token_data = []
    previous_hidden_state = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if past_key_values is None:
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
            else:
                last_token = generated_ids[:, -1:]
                outputs = model(
                    input_ids=last_token,
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True
                )

            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            last_layer_hidden = outputs.hidden_states[-1][:, -1, :].detach().cpu()
            last_layer_hidden_list = last_layer_hidden[0].tolist()

            sep_score = compute_sep_score(last_layer_hidden_list)
            hallushift_score = compute_hallushift_score(last_layer_hidden_list, previous_hidden_state)

            next_token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=False)

            token_data.append({
                "step": step,
                "token_id": int(next_token_id.item()),
                "token_text": next_token_text,
                "hidden_state": last_layer_hidden_list,
                "sep_score": sep_score,
                "hallushift_score": hallushift_score
            })

            generated_ids = torch.cat([generated_ids, next_token_id.to(device)], dim=1)
            previous_hidden_state = last_layer_hidden_list

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    generated_part = generated_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_part, skip_special_tokens=True).strip()

    return response, token_data


def generate_answer(prompt: str, max_new_tokens: int = 80) -> str:
    response, _ = generate_with_hidden_states(prompt, max_new_tokens=max_new_tokens)
    return response