from pathlib import Path
import json

INPUT_PATH = Path("data/labeled/ragtruth_normalized.jsonl")
OUTPUT_PATH = Path("data/processed/ragtruth_span_rows.jsonl")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    total_span_rows = 0
    examples_with_spans = 0
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

            spans = row.get("spans", [])

            if not spans:
                skipped_examples += 1
                continue

            examples_with_spans += 1

            for span_index, span in enumerate(spans):
                span_row = {
                    "example_id": example_id,
                    "span_index": span_index,
                    "span_text": span.get("span_text"),
                    "start_token": span.get("start_token"),
                    "end_token": span.get("end_token"),
                    "span_label": span.get("label"),
                    "prompt": prompt,
                    "response": response,
                    "task_type": task_type,
                    "source_name": source_name,
                    "model_name": model_name,
                    "quality": quality,
                    "split": split
                }

                fout.write(json.dumps(span_row, ensure_ascii=False) + "\n")
                total_span_rows += 1

    print("===== RAGTruth Span Row Build Complete =====")
    print(f"Total response examples read: {total_examples}")
    print(f"Examples with spans: {examples_with_spans}")
    print(f"Total span rows saved: {total_span_rows}")
    print(f"Skipped response examples with no spans: {skipped_examples}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
