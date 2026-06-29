from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


TRAIN_PATH = Path("data/splits/device_1_train_ordered_by_source.csv")
TEST_PATH = Path("data/splits/device_1_test_ordered_by_source.csv")
REPORT_DIR = Path("outputs/reports")
REPORT_PATH = REPORT_DIR / "device_1_ordered_random_forest_baseline.txt"
CONFUSION_MATRIX_PATH = REPORT_DIR / "device_1_ordered_random_forest_confusion_matrix.csv"
FEATURE_IMPORTANCE_PATH = REPORT_DIR / "device_1_ordered_random_forest_feature_importance.csv"

TARGET_COLUMN = "binary_target"
DROP_COLUMNS = ["binary_label", "binary_target", "source_file"]
RANDOM_STATE = 42


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=DROP_COLUMNS), frame[TARGET_COLUMN]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading ordered train/test CSV files...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print("Training Random Forest ordered baseline...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    train_start = perf_counter()
    model.fit(X_train, y_train)
    train_seconds = perf_counter() - train_start

    predict_start = perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = perf_counter() - predict_start

    accuracy = accuracy_score(y_test, y_pred)
    attack_precision, attack_recall, attack_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    benign_precision, benign_recall, benign_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=0, zero_division=0
    )
    report = classification_report(
        y_test, y_pred, target_names=["benign", "attack"], zero_division=0
    )
    matrix_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=[0, 1]),
        index=["actual_benign", "actual_attack"],
        columns=["predicted_benign", "predicted_attack"],
    )
    matrix_df.to_csv(CONFUSION_MATRIX_PATH)

    feature_importance = (
        pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    summary = f"""Device 1 Ordered Random Forest Baseline
=======================================

Dataset
-------
Split strategy: source_file ordered 80/20
Train rows: {len(train_df)}
Test rows: {len(test_df)}
Feature count: {X_train.shape[1]}

Model
-----
RandomForestClassifier
n_estimators: {model.n_estimators}
random_state: {RANDOM_STATE}
class balancing: none
feature selection: none
scaling: none

Timing
------
Train seconds: {train_seconds:.2f}
Predict seconds: {predict_seconds:.2f}

Metrics
-------
Accuracy: {accuracy:.6f}
Attack precision: {attack_precision:.6f}
Attack recall: {attack_recall:.6f}
Attack F1: {attack_f1:.6f}
Benign precision: {benign_precision:.6f}
Benign recall: {benign_recall:.6f}
Benign F1: {benign_f1:.6f}

Confusion Matrix
----------------
{matrix_df.to_string()}

Classification Report
---------------------
{report}

Top 20 Feature Importances
--------------------------
{feature_importance.head(20).to_string(index=False)}
"""
    REPORT_PATH.write_text(summary)

    print(summary)
    print(f"Report written to: {REPORT_PATH}")
    print(f"Confusion matrix written to: {CONFUSION_MATRIX_PATH}")
    print(f"Feature importance written to: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
