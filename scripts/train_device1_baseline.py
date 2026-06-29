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


TRAIN_PATH = Path("data/splits/device_1_train.csv")
TEST_PATH = Path("data/splits/device_1_test.csv")
REPORT_DIR = Path("outputs/reports")
REPORT_PATH = REPORT_DIR / "device_1_random_forest_baseline.txt"
CONFUSION_MATRIX_PATH = REPORT_DIR / "device_1_random_forest_confusion_matrix.csv"
FEATURE_IMPORTANCE_PATH = REPORT_DIR / "device_1_random_forest_feature_importance.csv"

TARGET_COLUMN = "binary_target"
DROP_COLUMNS = ["binary_label", "binary_target", "source_file"]
RANDOM_STATE = 42


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = frame.drop(columns=DROP_COLUMNS)
    y = frame[TARGET_COLUMN]
    return X, y


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing train CSV: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading train/test CSV files...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print("Training Random Forest baseline...")

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
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        pos_label=1,
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
                "feature": X_train.columns,
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

    summary = f"""Device 1 Random Forest Baseline
================================

Dataset
-------
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
Attack precision: {precision:.6f}
Attack recall: {recall:.6f}
Attack F1: {f1:.6f}

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
