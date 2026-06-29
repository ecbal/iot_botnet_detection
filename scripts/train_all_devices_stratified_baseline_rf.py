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


TRAIN_PATH = Path("data/splits/all_devices_train_stratified.csv")
TEST_PATH = Path("data/splits/all_devices_test_stratified.csv")
REPORT_DIR = Path("outputs/reports")
REPORT_PATH = REPORT_DIR / "all_devices_stratified_random_forest_baseline.txt"
CONFUSION_MATRIX_PATH = (
    REPORT_DIR / "all_devices_stratified_random_forest_confusion_matrix.csv"
)
FEATURE_IMPORTANCE_PATH = (
    REPORT_DIR / "all_devices_stratified_random_forest_feature_importance.csv"
)

TARGET_COLUMN = "binary_target"
DROP_COLUMNS = ["binary_label", "binary_target", "source_file"]
RANDOM_STATE = 42


def get_feature_columns() -> list[str]:
    columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
    return [column for column in columns if column not in DROP_COLUMNS]


def read_features_target(path: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    dtype = {column: "float32" for column in feature_columns}
    dtype[TARGET_COLUMN] = "uint8"

    frame = pd.read_csv(
        path,
        usecols=[*feature_columns, TARGET_COLUMN],
        dtype=dtype,
    )
    y = frame.pop(TARGET_COLUMN)
    return frame, y


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing train CSV: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = get_feature_columns()

    print("Reading train CSV...")
    read_train_start = perf_counter()
    X_train, y_train = read_features_target(TRAIN_PATH, feature_columns)
    read_train_seconds = perf_counter() - read_train_start

    print("Reading test CSV...")
    read_test_start = perf_counter()
    X_test, y_test = read_features_target(TEST_PATH, feature_columns)
    read_test_seconds = perf_counter() - read_test_start

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print("Training Random Forest baseline...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    train_start = perf_counter()
    model.fit(X_train, y_train)
    train_seconds = perf_counter() - train_start

    print("Predicting test split...")
    predict_start = perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = perf_counter() - predict_start

    accuracy = accuracy_score(y_test, y_pred)
    attack_precision, attack_recall, attack_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    benign_precision, benign_recall, benign_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        pos_label=0,
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=["benign", "attack"],
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_benign", "actual_attack"],
        columns=["predicted_benign", "predicted_attack"],
    )
    matrix_df.to_csv(CONFUSION_MATRIX_PATH)

    summary = f"""All Devices Stratified Random Forest Baseline
================================================

Dataset
-------
Train rows: {len(X_train)}
Test rows: {len(X_test)}
Feature count: {X_train.shape[1]}

Train distribution:
{y_train.value_counts().sort_index().to_string()}

Test distribution:
{y_test.value_counts().sort_index().to_string()}

Model
-----
RandomForestClassifier
n_estimators: {model.n_estimators}
random_state: {RANDOM_STATE}
class balancing: none
feature selection: none
scaling: none
split strategy: stratified 80/20 by binary_target

Timing
------
Read train seconds: {read_train_seconds:.2f}
Read test seconds: {read_test_seconds:.2f}
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
