from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/labeled_devices/device_1_labeled.csv")
OUTPUT_DIR = Path("data/splits")
TRAIN_PATH = OUTPUT_DIR / "device_1_train.csv"
TEST_PATH = OUTPUT_DIR / "device_1_test.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing labeled CSV: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["binary_target"],
        shuffle=True,
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Train: {TRAIN_PATH} rows={len(train_df)}")
    print(train_df["binary_label"].value_counts().to_string())
    print()
    print(f"Test: {TEST_PATH} rows={len(test_df)}")
    print(test_df["binary_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
