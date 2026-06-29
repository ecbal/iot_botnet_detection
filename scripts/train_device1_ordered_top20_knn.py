from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


TRAIN_PATH = Path("data/splits/device_1_train_ordered_top20.csv")
TEST_PATH = Path("data/splits/device_1_test_ordered_top20.csv")
REPORT_DIR = Path("outputs/reports")
REPORT_PATH = REPORT_DIR / "device_1_ordered_knn_top20.txt"
CONFUSION_MATRIX_PATH = REPORT_DIR / "device_1_ordered_knn_top20_confusion_matrix.csv"

TARGET_COLUMN = "binary_target"
DROP_COLUMNS = ["binary_label", "binary_target", "source_file"]
N_NEIGHBORS = 5
PREDICT_BATCH_SIZE = 5_000


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=DROP_COLUMNS), frame[TARGET_COLUMN]


def predict_in_batches(model: KNeighborsClassifier, X_test: np.ndarray) -> np.ndarray:
    predictions = []
    for start in range(0, len(X_test), PREDICT_BATCH_SIZE):
        end = min(start + PREDICT_BATCH_SIZE, len(X_test))
        predictions.append(model.predict(X_test[start:end]))
        print(f"Predicted rows {end}/{len(X_test)}")
    return np.concatenate(predictions)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading ordered top20 train/test CSV files...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print("Scaling features with StandardScaler fit on train only...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training KNN ordered top20...")
    model = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        weights="distance",
        algorithm="auto",
        n_jobs=-1,
    )

    train_start = perf_counter()
    model.fit(X_train_scaled, y_train)
    train_seconds = perf_counter() - train_start

    predict_start = perf_counter()
    y_pred = predict_in_batches(model, X_test_scaled)
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

    summary = f"""Device 1 Ordered KNN Top20
===========================

Dataset
-------
Split strategy: source_file ordered 80/20
Train rows: {len(train_df)}
Test rows: {len(test_df)}
Feature count: {X_train.shape[1]}

Model
-----
KNeighborsClassifier
n_neighbors: {N_NEIGHBORS}
weights: distance
class balancing: none
feature selection: ordered RF feature importance top20
scaling: StandardScaler fit on train only

Timing
------
Fit seconds: {train_seconds:.2f}
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

Used Features
-------------
{chr(10).join(X_train.columns)}
"""
    REPORT_PATH.write_text(summary)

    print(summary)
    print(f"Report written to: {REPORT_PATH}")
    print(f"Confusion matrix written to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
