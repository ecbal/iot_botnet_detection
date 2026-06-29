from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/labeled_devices/all_devices_labeled.csv")
OUTPUT_DIR = Path("data/splits")
TRAIN_PATH = OUTPUT_DIR / "all_devices_train_stratified.csv"
TEST_PATH = OUTPUT_DIR / "all_devices_test_stratified.csv"
TARGET_COLUMN = "binary_target"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CHUNKSIZE = 100_000


def count_targets() -> Counter:
    counts = Counter()
    for chunk in pd.read_csv(INPUT_PATH, usecols=[TARGET_COLUMN], chunksize=CHUNKSIZE):
        counts.update(chunk[TARGET_COLUMN].value_counts().to_dict())
    return counts


def calculate_test_counts(target_counts: Counter) -> dict[int, int]:
    return {
        target: int(round(count * TEST_SIZE))
        for target, count in target_counts.items()
    }


def split_chunk(
    chunk: pd.DataFrame,
    rows_remaining: Counter,
    test_remaining: Counter,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = np.zeros(len(chunk), dtype=bool)

    for idx, target in enumerate(chunk[TARGET_COLUMN].to_numpy()):
        probability = test_remaining[target] / rows_remaining[target]
        if rng.random() < probability:
            test_mask[idx] = True
            test_remaining[target] -= 1
        rows_remaining[target] -= 1

    return chunk.loc[~test_mask], chunk.loc[test_mask]


def write_split() -> tuple[int, int, Counter, Counter]:
    target_counts = count_targets()
    rows_remaining = Counter(target_counts)
    test_remaining = Counter(calculate_test_counts(target_counts))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [TRAIN_PATH, TEST_PATH]:
        if path.exists():
            path.unlink()

    rng = np.random.default_rng(RANDOM_STATE)
    train_rows = 0
    test_rows = 0
    train_counts = Counter()
    test_counts = Counter()
    write_train_header = True
    write_test_header = True

    for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE):
        train_chunk, test_chunk = split_chunk(
            chunk,
            rows_remaining,
            test_remaining,
            rng,
        )

        if not train_chunk.empty:
            train_chunk.to_csv(
                TRAIN_PATH,
                mode="w" if write_train_header else "a",
                header=write_train_header,
                index=False,
            )
            train_rows += len(train_chunk)
            train_counts.update(train_chunk[TARGET_COLUMN].value_counts().to_dict())
            write_train_header = False

        if not test_chunk.empty:
            test_chunk.to_csv(
                TEST_PATH,
                mode="w" if write_test_header else "a",
                header=write_test_header,
                index=False,
            )
            test_rows += len(test_chunk)
            test_counts.update(test_chunk[TARGET_COLUMN].value_counts().to_dict())
            write_test_header = False

    return train_rows, test_rows, train_counts, test_counts


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing labeled CSV: {INPUT_PATH}")

    train_rows, test_rows, train_counts, test_counts = write_split()

    print(f"Train: {TRAIN_PATH} rows={train_rows}")
    print(pd.Series(train_counts).sort_index().to_string())
    print()
    print(f"Test: {TEST_PATH} rows={test_rows}")
    print(pd.Series(test_counts).sort_index().to_string())


if __name__ == "__main__":
    main()
