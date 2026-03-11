from pathlib import Path
import json

INPUT_PATH = Path("data/labeled/ragtruth_normalized.jsonl")
OUTPUT_PATH = Path("data/processed/ragtruth_token_rows.jsonl")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    total_token_rows = 0
    hallucinated_token_rows = 0
    skipped_examples = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            total_examples += 1

            example_id = row.get("id")
            prompt = row.get("prompt")
            response = row.get("response")
            task_type = row.get("task_type")
            metadata = row.get("metadata", {})

            source_name = metadata.get("source_name")
            model_name = metadata.get("model")
            quality = metadata.get("quality")
            split = metadata.get("split")

            tokens = row.get("tokens", [])
            token_labels = row.get("token_labels", [])

            if not tokens or not token_labels:
                skipped_examples += 1
                continue

            if len(tokens) != len(token_labels):
                skipped_examples += 1
                continue

            for token_index, (token_text, token_label) in enumerate(zip(tokens, token_labels)):
                token_row = {
                    "example_id": example_id,
                    "token_index": token_index,
                    "token_text": token_text,
                    "token_label": int(token_label),
                    "prompt": prompt,
                    "response": response,
                    "task_type": task_type,
                    "source_name": source_name,
                    "model_name": model_name,
                    "quality": quality,
                    "split": split
                }

                fout.write(json.dumps(token_row, ensure_ascii=False) + "\n")
                total_token_rows += 1

                if int(token_label) == 1:
                    hallucinated_token_rows += 1

    print("===== RAGTruth Token Row Build Complete =====")
    print(f"Total response examples read: {total_examples}")
    print(f"Total token rows saved: {total_token_rows}")
    print(f"Hallucinated token rows: {hallucinated_token_rows}")
    print(f"Skipped response examples: {skipped_examples}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()