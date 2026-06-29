from pathlib import Path

import pandas as pd


TRAIN_PATH = Path("data/splits/device_1_train_ordered_by_source.csv")
TEST_PATH = Path("data/splits/device_1_test_ordered_by_source.csv")
FEATURE_IMPORTANCE_PATH = Path("outputs/reports/device_1_ordered_random_forest_feature_importance.csv")
TRAIN_TOP20_PATH = Path("data/splits/device_1_train_ordered_top20.csv")
TEST_TOP20_PATH = Path("data/splits/device_1_test_ordered_top20.csv")

LABEL_COLUMNS = ["binary_label", "binary_target", "source_file"]
TOP_N = 20


def main() -> None:
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    top_features = feature_importance.head(TOP_N)["feature"].tolist()
    selected_columns = top_features + LABEL_COLUMNS

    train_df = pd.read_csv(TRAIN_PATH, usecols=selected_columns)
    test_df = pd.read_csv(TEST_PATH, usecols=selected_columns)

    train_df.to_csv(TRAIN_TOP20_PATH, index=False)
    test_df.to_csv(TEST_TOP20_PATH, index=False)

    print("Ordered top 20 features:")
    for index, feature in enumerate(top_features, start=1):
        print(f"{index:02d}. {feature}")
    print()
    print(f"Train top20 written to: {TRAIN_TOP20_PATH}")
    print(f"Test top20 written to: {TEST_TOP20_PATH}")


if __name__ == "__main__":
    main()
