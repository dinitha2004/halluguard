from pathlib import Path
import json

import numpy as np

INPUT_PATH = Path("data/features/ragtruth_hidden_state_rows_pilot_20.jsonl")
OUTPUT_PATH = Path("data/features/ragtruth_token_features_pilot_20.jsonl")


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_example_lengths(rows):
    example_lengths = {}

    for row in rows:
        example_id = row["example_id"]
        token_index = int(row["token_index"])

        if example_id not in example_lengths:
            example_lengths[example_id] = token_index
        else:
            example_lengths[example_id] = max(example_lengths[example_id], token_index)

    for example_id in example_lengths:
        example_lengths[example_id] = example_lengths[example_id] + 1

    return example_lengths


def safe_position_ratio(token_index: int, total_tokens: int):
    if total_tokens <= 1:
        return 0.0
    return float(token_index) / float(total_tokens - 1)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(INPUT_PATH)
    example_lengths = build_example_lengths(rows)

    total_rows = 0
    positive_rows = 0
    hidden_dim_seen = None

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for row in rows:
            example_id = row["example_id"]
            token_index = int(row["token_index"])
            token_text = row["token_text"]
            label = int(row["token_label"])
            matched_span_label = row.get("matched_span_label", "NONE")

            hidden = np.array(row["hidden_state"], dtype=np.float32)
            total_tokens = example_lengths[example_id]

            feature_row = {
                "example_id": example_id,
                "token_index": token_index,
                "token_text": token_text,
                "label": label,
                "matched_span_label": matched_span_label,
                "task_type": row.get("task_type"),
                "source_name": row.get("source_name"),
                "model_name": row.get("model_name"),
                "quality": row.get("quality"),
                "split": row.get("split"),
                "char_start": row.get("char_start"),
                "char_end": row.get("char_end"),
                "total_tokens_in_example": total_tokens,
                "position_ratio": safe_position_ratio(token_index, total_tokens),
                "hidden_dim": int(hidden.shape[0]),
                "hidden_norm_l2": float(np.linalg.norm(hidden)),
                "hidden_mean": float(hidden.mean()),
                "hidden_std": float(hidden.std()),
                "hidden_min": float(hidden.min()),
                "hidden_max": float(hidden.max()),
                "hidden_abs_mean": float(np.mean(np.abs(hidden))),
                "hidden_abs_max": float(np.max(np.abs(hidden))),
                "hidden_state": hidden.tolist(),
            }

            fout.write(json.dumps(feature_row, ensure_ascii=False) + "\n")

            total_rows += 1
            if label == 1:
                positive_rows += 1

            if hidden_dim_seen is None:
                hidden_dim_seen = int(hidden.shape[0])

    print("===== Token Feature Build Complete =====")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Total token rows written: {total_rows}")
    print(f"Positive label rows: {positive_rows}")
    print(f"Hidden dimension: {hidden_dim_seen}")


if __name__ == "__main__":
    main()