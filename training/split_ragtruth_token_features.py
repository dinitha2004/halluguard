from pathlib import Path
import json
import random

INPUT_PATH = Path("data/features/ragtruth_token_features_pilot_20.jsonl")

TRAIN_PATH = Path("data/splits/train_features.jsonl")
VAL_PATH = Path("data/splits/val_features.jsonl")
TEST_PATH = Path("data/splits/test_features.jsonl")

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def group_rows_by_example(rows):
    grouped = {}

    for row in rows:
        example_id = row["example_id"]
        if example_id not in grouped:
            grouped[example_id] = []
        grouped[example_id].append(row)

    return grouped


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_positive_labels(rows):
    return sum(1 for row in rows if int(row["label"]) == 1)


def main():
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must add up to 1.0")

    rows = load_rows(INPUT_PATH)
    grouped = group_rows_by_example(rows)

    example_ids = list(grouped.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(example_ids)

    total_examples = len(example_ids)

    train_end = int(total_examples * TRAIN_RATIO)
    val_end = train_end + int(total_examples * VAL_RATIO)

    train_example_ids = example_ids[:train_end]
    val_example_ids = example_ids[train_end:val_end]
    test_example_ids = example_ids[val_end:]

    train_rows = []
    val_rows = []
    test_rows = []

    for example_id in train_example_ids:
        train_rows.extend(grouped[example_id])

    for example_id in val_example_ids:
        val_rows.extend(grouped[example_id])

    for example_id in test_example_ids:
        test_rows.extend(grouped[example_id])

    write_rows(TRAIN_PATH, train_rows)
    write_rows(VAL_PATH, val_rows)
    write_rows(TEST_PATH, test_rows)

    print("===== Feature Split Complete =====")
    print(f"Input file: {INPUT_PATH}")
    print(f"Total examples: {total_examples}")
    print(f"Total token rows: {len(rows)}")
    print()

    print("Train split")
    print(f"  Example count: {len(train_example_ids)}")
    print(f"  Token row count: {len(train_rows)}")
    print(f"  Positive labels: {count_positive_labels(train_rows)}")
    print(f"  Output: {TRAIN_PATH}")
    print()

    print("Validation split")
    print(f"  Example count: {len(val_example_ids)}")
    print(f"  Token row count: {len(val_rows)}")
    print(f"  Positive labels: {count_positive_labels(val_rows)}")
    print(f"  Output: {VAL_PATH}")
    print()

    print("Test split")
    print(f"  Example count: {len(test_example_ids)}")
    print(f"  Token row count: {len(test_rows)}")
    print(f"  Positive labels: {count_positive_labels(test_rows)}")
    print(f"  Output: {TEST_PATH}")


if __name__ == "__main__":
    main()