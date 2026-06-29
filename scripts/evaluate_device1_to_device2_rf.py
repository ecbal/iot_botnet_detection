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


DEVICE1_TRAIN_115 = Path("data/splits/device_1_train_ordered_by_source.csv")
DEVICE1_TRAIN_TOP20 = Path("data/splits/device_1_train_ordered_top20.csv")
DEVICE2_TEST = Path("data/labeled_devices/device_2_labeled.csv")
FEATURE_IMPORTANCE_PATH = Path("outputs/reports/device_1_ordered_random_forest_feature_importance.csv")
REPORT_DIR = Path("outputs/reports")
REPORT_PATH = REPORT_DIR / "device_1_train_device_2_test_rf_comparison.txt"

TARGET_COLUMN = "binary_target"
DROP_COLUMNS = ["binary_label", "binary_target", "source_file"]
LABELS = [0, 1]
RANDOM_STATE = 42
TOP_N = 20


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=DROP_COLUMNS), frame[TARGET_COLUMN]


def load_top_features() -> list[str]:
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return feature_importance.head(TOP_N)["feature"].tolist()


def evaluate_predictions(name: str, y_test: pd.Series, y_pred) -> str:
    accuracy = accuracy_score(y_test, y_pred)
    attack_precision, attack_recall, attack_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    benign_precision, benign_recall, benign_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=0, zero_division=0
    )
    matrix_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=LABELS),
        index=["actual_benign", "actual_attack"],
        columns=["predicted_benign", "predicted_attack"],
    )
    report = classification_report(
        y_test, y_pred, target_names=["benign", "attack"], zero_division=0
    )

    matrix_path = REPORT_DIR / f"{name}_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    return f"""Model: {name}
{'-' * (7 + len(name))}
Accuracy: {accuracy:.6f}
Attack precision: {attack_precision:.6f}
Attack recall: {attack_recall:.6f}
Attack F1: {attack_f1:.6f}
Benign precision: {benign_precision:.6f}
Benign recall: {benign_recall:.6f}
Benign F1: {benign_f1:.6f}

Confusion Matrix:
{matrix_df.to_string()}

Classification Report:
{report}
Confusion matrix CSV: {matrix_path}
"""


def train_predict_rf(X_train, y_train, X_test) -> tuple[object, float, float]:
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
    return y_pred, train_seconds, predict_seconds


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading device 1 ordered train 115...")
    train_115 = pd.read_csv(DEVICE1_TRAIN_115)
    print("Reading device 2 full labeled test...")
    test_full = pd.read_csv(DEVICE2_TEST)

    X_train_115, y_train = split_features_target(train_115)
    X_test_115, y_test = split_features_target(test_full)

    print("Training RF 115 on device 1, testing on device 2...")
    y_pred_115, train_115_seconds, predict_115_seconds = train_predict_rf(
        X_train_115,
        y_train,
        X_test_115,
    )
    report_115 = evaluate_predictions("device1_to_device2_rf_115", y_test, y_pred_115)

    top_features = load_top_features()
    print("Training RF top20 on device 1, testing on device 2...")
    X_train_top20 = train_115[top_features]
    X_test_top20 = test_full[top_features]
    y_pred_top20, train_top20_seconds, predict_top20_seconds = train_predict_rf(
        X_train_top20,
        y_train,
        X_test_top20,
    )
    report_top20 = evaluate_predictions("device1_to_device2_rf_top20", y_test, y_pred_top20)

    print("Training RF top20 + SMOTE on device 1, testing on device 2...")
    scaler = StandardScaler()
    X_train_top20_scaled = scaler.fit_transform(X_train_top20)
    X_test_top20_scaled = scaler.transform(X_test_top20)

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    smote_start = perf_counter()
    X_train_smote, y_train_smote = smote.fit_resample(X_train_top20_scaled, y_train)
    smote_seconds = perf_counter() - smote_start

    y_pred_smote, train_smote_seconds, predict_smote_seconds = train_predict_rf(
        X_train_smote,
        y_train_smote,
        X_test_top20_scaled,
    )
    report_smote = evaluate_predictions("device1_to_device2_rf_top20_smote", y_test, y_pred_smote)

    summary = f"""Device 1 Train -> Device 2 Full Test
====================================

Goal
----
Train on device 1 ordered split train set, then test on all labeled device 2 rows without splitting device 2.

Device 1 train rows: {len(train_115)}
Device 2 test rows: {len(test_full)}

Device 1 train distribution:
{y_train.value_counts().sort_index().to_string()}

Device 2 test distribution:
{y_test.value_counts().sort_index().to_string()}

Timing
------
RF 115 train seconds: {train_115_seconds:.2f}
RF 115 predict seconds: {predict_115_seconds:.2f}

RF top20 train seconds: {train_top20_seconds:.2f}
RF top20 predict seconds: {predict_top20_seconds:.2f}

RF top20 + SMOTE seconds: {smote_seconds:.2f}
RF top20 + SMOTE train seconds: {train_smote_seconds:.2f}
RF top20 + SMOTE predict seconds: {predict_smote_seconds:.2f}

Top20 Features
--------------
{chr(10).join(top_features)}

Results
-------
{report_115}

{report_top20}

{report_smote}
"""

    REPORT_PATH.write_text(summary)
    print(summary)
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
