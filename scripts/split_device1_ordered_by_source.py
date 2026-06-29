from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/labeled_devices/device_1_labeled.csv")
OUTPUT_DIR = Path("data/splits")
TRAIN_PATH = OUTPUT_DIR / "device_1_train_ordered_by_source.csv"
TEST_PATH = OUTPUT_DIR / "device_1_test_ordered_by_source.csv"
TRAIN_RATIO = 0.8


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing labeled CSV: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading labeled CSV: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    train_parts = []
    test_parts = []
    split_rows = []

    for source_file, group in df.groupby("source_file", sort=False):
        split_index = int(len(group) * TRAIN_RATIO)
        train_part = group.iloc[:split_index]
        test_part = group.iloc[split_index:]

        train_parts.append(train_part)
        test_parts.append(test_part)
        split_rows.append(
            {
                "source_file": source_file,
                "total_rows": len(group),
                "train_rows": len(train_part),
                "test_rows": len(test_part),
                "binary_label": group["binary_label"].iloc[0],
            }
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    print(f"Writing train CSV: {TRAIN_PATH}")
    train_df.to_csv(TRAIN_PATH, index=False)

    print(f"Writing test CSV: {TEST_PATH}")
    test_df.to_csv(TEST_PATH, index=False)

    split_summary = pd.DataFrame(split_rows)

    print()
    print("Split by source_file:")
    print(split_summary.to_string(index=False))
    print()
    print(f"Train rows: {len(train_df)}")
    print(train_df["binary_label"].value_counts().to_string())
    print()
    print(f"Test rows: {len(test_df)}")
    print(test_df["binary_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
