from pathlib import Path
from time import perf_counter

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler


TRAIN_PATH = Path("data/splits/device_1_train_ordered_top20.csv")
TEST_PATH = Path("data/splits/device_1_test_ordered_top20.csv")
REPORT_DIR = Path("outputs/reports")
REPORT_PATH = REPORT_DIR / "device_1_ordered_random_forest_top20_smote.txt"
CONFUSION_MATRIX_PATH = REPORT_DIR / "device_1_ordered_random_forest_top20_smote_confusion_matrix.csv"

TARGET_COLUMN = "binary_target"
DROP_COLUMNS = ["binary_label", "binary_target", "source_file"]
RANDOM_STATE = 42


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=DROP_COLUMNS), frame[TARGET_COLUMN]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading ordered top20 train/test CSV files...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    print(f"Train shape before SMOTE: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print("Train class distribution before SMOTE:")
    print(y_train.value_counts().sort_index().to_string())

    print("Scaling features with StandardScaler fit on train only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Applying SMOTE to ordered train only...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    smote_start = perf_counter()
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    smote_seconds = perf_counter() - smote_start

    print(f"Train shape after SMOTE: {X_train_smote.shape}")
    print("Train class distribution after SMOTE:")
    print(pd.Series(y_train_smote).value_counts().sort_index().to_string())

    print("Training Random Forest ordered top20 + SMOTE...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    train_start = perf_counter()
    model.fit(X_train_smote, y_train_smote)
    train_seconds = perf_counter() - train_start

    predict_start = perf_counter()
    y_pred = model.predict(X_test_scaled)
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

    before_counts = y_train.value_counts().sort_index()
    after_counts = pd.Series(y_train_smote).value_counts().sort_index()

    summary = f"""Device 1 Ordered Random Forest Top20 + SMOTE
===============================================

Dataset
-------
Split strategy: source_file ordered 80/20
Train rows before SMOTE: {len(train_df)}
Train rows after SMOTE: {len(y_train_smote)}
Test rows: {len(test_df)}
Feature count: {X_train.shape[1]}

Train Distribution Before SMOTE
-------------------------------
benign: {before_counts.get(0, 0)}
attack: {before_counts.get(1, 0)}

Train Distribution After SMOTE
------------------------------
benign: {after_counts.get(0, 0)}
attack: {after_counts.get(1, 0)}

Model
-----
RandomForestClassifier
n_estimators: {model.n_estimators}
random_state: {RANDOM_STATE}
class balancing: SMOTE on train only
feature selection: ordered RF feature importance top20
scaling: StandardScaler fit on train only

Timing
------
SMOTE seconds: {smote_seconds:.2f}
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
