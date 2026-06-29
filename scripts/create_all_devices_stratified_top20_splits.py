from pathlib import Path

import pandas as pd


TRAIN_PATH = Path("data/splits/all_devices_train_stratified.csv")
TEST_PATH = Path("data/splits/all_devices_test_stratified.csv")
FEATURE_IMPORTANCE_PATH = (
    Path("outputs/reports")
    / "all_devices_stratified_random_forest_feature_importance.csv"
)
OUTPUT_DIR = Path("data/splits")
TRAIN_TOP20_PATH = OUTPUT_DIR / "all_devices_train_stratified_top20.csv"
TEST_TOP20_PATH = OUTPUT_DIR / "all_devices_test_stratified_top20.csv"

LABEL_COLUMNS = ["binary_label", "binary_target", "source_file"]
TOP_N = 20
CHUNKSIZE = 100_000


def load_top_features() -> list[str]:
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return feature_importance.head(TOP_N)["feature"].tolist()


def write_selected_columns(input_path: Path, output_path: Path, columns: list[str]) -> int:
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    write_header = True

    for chunk in pd.read_csv(input_path, usecols=columns, chunksize=CHUNKSIZE):
        chunk.to_csv(
            output_path,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
        )
        total_rows += len(chunk)
        write_header = False

    return total_rows


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing train CSV: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_PATH}")
    if not FEATURE_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(f"Missing feature importance CSV: {FEATURE_IMPORTANCE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    top_features = load_top_features()
    selected_columns = top_features + LABEL_COLUMNS

    train_rows = write_selected_columns(TRAIN_PATH, TRAIN_TOP20_PATH, selected_columns)
    test_rows = write_selected_columns(TEST_PATH, TEST_TOP20_PATH, selected_columns)

    print("Top 20 features:")
    for index, feature in enumerate(top_features, start=1):
        print(f"{index:02d}. {feature}")

    print()
    print(f"Train top20 written to: {TRAIN_TOP20_PATH} rows={train_rows}")
    print(f"Test top20 written to: {TEST_TOP20_PATH} rows={test_rows}")


if __name__ == "__main__":
    main()
