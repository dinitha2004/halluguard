from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.load_model import MODEL_NAME, get_device

INPUT_PATH = Path("data/labeled/ragtruth_normalized.jsonl")
OUTPUT_PATH = Path("data/features/ragtruth_hidden_state_rows_pilot_20.jsonl")

LIMIT_EXAMPLES = 20
MAX_LENGTH = 512


def get_dtype(device: str):
    if device in ("mps", "cuda"):
        return torch.float16
    return torch.float32


def load_base_model():
    device = get_device()
    dtype = get_dtype(device)

    print(f"Loading tokenizer for model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model on device: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


def find_span_ranges_in_response(response: str, spans: list):
    """
    Convert span_text entries into character ranges inside the response.
    This is a simple sequential search approach.
    """
    ranges = []
    search_start = 0

    for span in spans:
        span_text = span.get("span_text")
        span_label = span.get("label")

        if not span_text:
            continue

        idx = response.find(span_text, search_start)

        if idx == -1:
            idx = response.find(span_text)

        if idx == -1:
            continue

        start_char = idx
        end_char = idx + len(span_text)

        ranges.append(
            {
                "span_text": span_text,
                "span_label": span_label,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

        search_start = end_char

    return ranges


def token_overlaps_any_span(token_start, token_end, span_ranges):
    """
    Return:
    - token_label: 0 or 1
    - matched_span_label: LOW / MEDIUM / HIGH / NONE
    """
    if token_start == token_end:
        return 0, "NONE"

    matched_labels = []

    for span in span_ranges:
        span_start = span["start_char"]
        span_end = span["end_char"]

        overlap = max(token_start, span_start) < min(token_end, span_end)
        if overlap:
            matched_labels.append(span["span_label"])

    if not matched_labels:
        return 0, "NONE"

    if "HIGH" in matched_labels:
        return 1, "HIGH"
    if "MEDIUM" in matched_labels:
        return 1, "MEDIUM"
    return 1, "LOW"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_base_model()

    total_examples_read = 0
    total_examples_processed = 0
    total_token_rows_saved = 0
    skipped_empty_response = 0
    skipped_tokenizer_fail = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            if total_examples_processed >= LIMIT_EXAMPLES:
                break

            row = json.loads(line)
            total_examples_read += 1

            example_id = row.get("id")
            prompt = row.get("prompt")
            response = row.get("response")
            task_type = row.get("task_type")
            spans = row.get("spans", [])
            metadata = row.get("metadata", {})

            source_name = metadata.get("source_name")
            model_name = metadata.get("model")
            quality = metadata.get("quality")
            split = metadata.get("split")

            if not response or not response.strip():
                skipped_empty_response += 1
                continue

            span_ranges = find_span_ranges_in_response(response, spans)

            try:
                encoded = tokenizer(
                    response,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    add_special_tokens=False,
                )
            except Exception as e:
                print(f"Tokenizer failed for example {example_id}: {e}")
                skipped_tokenizer_fail += 1
                continue

            offset_mapping = encoded.pop("offset_mapping")[0].tolist()
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            last_hidden = outputs.hidden_states[-1][0].detach().cpu()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids[0])

            for token_index, (token_text, offsets) in enumerate(zip(token_texts, offset_mapping)):
                token_start, token_end = offsets
                token_label, matched_span_label = token_overlaps_any_span(
                    token_start,
                    token_end,
                    span_ranges,
                )

                hidden_state = [float(x) for x in last_hidden[token_index].tolist()]

                out_row = {
                    "example_id": example_id,
                    "token_index": token_index,
                    "token_text": token_text,
                    "token_label": token_label,
                    "matched_span_label": matched_span_label,
                    "char_start": token_start,
                    "char_end": token_end,
                    "prompt": prompt,
                    "response": response,
                    "task_type": task_type,
                    "source_name": source_name,
                    "model_name": model_name,
                    "quality": quality,
                    "split": split,
                    "hidden_state": hidden_state,
                }

                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                total_token_rows_saved += 1

            total_examples_processed += 1
            print(f"Processed example {total_examples_processed}/{LIMIT_EXAMPLES}: {example_id}")

    print("\n===== Hidden-State Extraction Pilot Complete =====")
    print(f"Total examples read: {total_examples_read}")
    print(f"Total examples processed: {total_examples_processed}")
    print(f"Total hidden-state token rows saved: {total_token_rows_saved}")
    print(f"Skipped empty responses: {skipped_empty_response}")
    print(f"Skipped tokenizer failures: {skipped_tokenizer_fail}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()