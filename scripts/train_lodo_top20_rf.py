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


LABELED_DIR = Path("data/labeled_devices")
DEVICE_INFO_PATH = Path("archive-2/device_info.csv")
FEATURE_IMPORTANCE_PATH = (
    Path("outputs/reports")
    / "all_devices_stratified_random_forest_feature_importance.csv"
)
TOP20_DEVICE_DIR = Path("data/lodo_top20")
REPORT_DIR = Path("outputs/reports")
RESULTS_CSV_PATH = REPORT_DIR / "lodo_top20_random_forest_results.csv"
SUMMARY_REPORT_PATH = REPORT_DIR / "lodo_top20_random_forest_report.md"
CONFUSION_DIR = REPORT_DIR / "lodo_top20_confusion_matrices"

LABEL_COLUMNS = ["binary_label", "binary_target", "source_file"]
TARGET_COLUMN = "binary_target"
DROP_COLUMNS = ["binary_label", "binary_target", "source_file"]
TOP_N = 20
RANDOM_STATE = 42
CHUNKSIZE = 100_000


def load_device_info() -> dict[int, str]:
    frame = pd.read_csv(DEVICE_INFO_PATH)
    return dict(zip(frame["DeviceID"], frame["DeviceName"], strict=True))


def load_top_features() -> list[str]:
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return feature_importance.head(TOP_N)["feature"].tolist()


def create_device_top20_csv(device_id: int, selected_columns: list[str]) -> tuple[Path, int]:
    input_path = LABELED_DIR / f"device_{device_id}_labeled.csv"
    output_path = TOP20_DEVICE_DIR / f"device_{device_id}_top20.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing labeled CSV: {input_path}")

    if output_path.exists():
        rows = sum(1 for _ in output_path.open()) - 1
        return output_path, rows

    total_rows = 0
    write_header = True
    for chunk in pd.read_csv(input_path, usecols=selected_columns, chunksize=CHUNKSIZE):
        chunk.to_csv(
            output_path,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
        )
        total_rows += len(chunk)
        write_header = False

    return output_path, total_rows


def ensure_top20_device_csvs(device_ids: list[int], selected_columns: list[str]) -> dict[int, Path]:
    TOP20_DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for device_id in device_ids:
        path, rows = create_device_top20_csv(device_id, selected_columns)
        paths[device_id] = path
        print(f"Top20 device cache: {path} rows={rows}")
    return paths


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


def read_train_devices(
    device_paths: dict[int, Path],
    train_device_ids: list[int],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    features = []
    targets = []
    for device_id in train_device_ids:
        X_device, y_device = read_features_target(device_paths[device_id], feature_columns)
        features.append(X_device)
        targets.append(y_device)

    X_train = pd.concat(features, ignore_index=True)
    y_train = pd.concat(targets, ignore_index=True)
    return X_train, y_train


def evaluate_fold(
    held_out_device_id: int,
    held_out_device_name: str,
    device_paths: dict[int, Path],
    device_ids: list[int],
    feature_columns: list[str],
) -> dict[str, object]:
    train_device_ids = [device_id for device_id in device_ids if device_id != held_out_device_id]
    test_path = device_paths[held_out_device_id]

    print()
    print(f"LODO fold: hold out device {held_out_device_id} ({held_out_device_name})")
    print(f"Train devices: {train_device_ids}")

    read_train_start = perf_counter()
    X_train, y_train = read_train_devices(device_paths, train_device_ids, feature_columns)
    read_train_seconds = perf_counter() - read_train_start

    read_test_start = perf_counter()
    X_test, y_test = read_features_target(test_path, feature_columns)
    read_test_seconds = perf_counter() - read_test_start

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    train_start = perf_counter()
    model.fit(X_train, y_train)
    train_seconds = perf_counter() - train_start

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
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_benign", "actual_attack"],
        columns=["predicted_benign", "predicted_attack"],
    )
    matrix_path = CONFUSION_DIR / f"device_{held_out_device_id}_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    report_path = CONFUSION_DIR / f"device_{held_out_device_id}_classification_report.txt"
    report_path.write_text(
        classification_report(
            y_test,
            y_pred,
            target_names=["benign", "attack"],
            zero_division=0,
        )
    )

    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    result = {
        "held_out_device_id": held_out_device_id,
        "held_out_device_name": held_out_device_name,
        "train_devices": " ".join(str(device_id) for device_id in train_device_ids),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_benign": int(train_counts.get(0, 0)),
        "train_attack": int(train_counts.get(1, 0)),
        "test_benign": int(test_counts.get(0, 0)),
        "test_attack": int(test_counts.get(1, 0)),
        "accuracy": accuracy,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
        "benign_precision": benign_precision,
        "benign_recall": benign_recall,
        "benign_f1": benign_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "true_benign_pred_benign": int(tn),
        "true_benign_pred_attack": int(fp),
        "true_attack_pred_benign": int(fn),
        "true_attack_pred_attack": int(tp),
        "read_train_seconds": read_train_seconds,
        "read_test_seconds": read_test_seconds,
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "confusion_matrix_path": str(matrix_path),
        "classification_report_path": str(report_path),
    }

    print(
        "Fold result: "
        f"accuracy={accuracy:.6f}, "
        f"attack_recall={attack_recall:.6f}, "
        f"false_negatives={fn}, "
        f"false_positives={fp}"
    )
    return result


def format_metric(value: float) -> str:
    return f"{value:.6f}"


def write_markdown_report(
    results: pd.DataFrame,
    device_info: dict[int, str],
    feature_columns: list[str],
) -> None:
    total_test_rows = int(results["test_rows"].sum())
    total_false_positives = int(results["true_benign_pred_attack"].sum())
    total_false_negatives = int(results["true_attack_pred_benign"].sum())
    total_true_positives = int(results["true_attack_pred_attack"].sum())
    total_true_negatives = int(results["true_benign_pred_benign"].sum())
    weighted_attack_recall = total_true_positives / (
        total_true_positives + total_false_negatives
    )
    weighted_benign_recall = total_true_negatives / (
        total_true_negatives + total_false_positives
    )
    overall_accuracy = (total_true_positives + total_true_negatives) / total_test_rows

    lines = [
        "# LODO Top20 Random Forest Report",
        "",
        "## Executive Summary",
        "",
        "This report evaluates strict cross-device generalization with a Leave-One-Device-Out (LODO) setup.",
        "",
        "Each fold holds out one IoT device completely as the test set and trains on the remaining eight devices. This is stricter than the earlier stratified random split because the held-out device contributes no rows to training.",
        "",
        "The model uses the same top20 feature set selected from the all-device 115-feature Random Forest baseline.",
        "",
        "## Experiment Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Task | Binary classification: benign vs attack |",
        "| Validation strategy | Leave-One-Device-Out |",
        "| Number of folds | 9 |",
        "| Model | RandomForestClassifier |",
        "| Estimators | 100 |",
        "| Random state | 42 |",
        "| Features | 20 RF-importance-selected features |",
        "| Scaling | none |",
        "| Balancing | none |",
        "",
        "## Devices",
        "",
        "| Device ID | Device Name |",
        "|---:|---|",
    ]

    for device_id, device_name in sorted(device_info.items()):
        lines.append(f"| {device_id} | {device_name} |")

    lines.extend(
        [
            "",
            "## Overall LODO Result",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total held-out test rows | {total_test_rows:,} |",
            f"| Overall accuracy from pooled confusion counts | {overall_accuracy:.6f} |",
            f"| Weighted attack recall from pooled counts | {weighted_attack_recall:.6f} |",
            f"| Weighted benign recall from pooled counts | {weighted_benign_recall:.6f} |",
            f"| Total false positives | {total_false_positives:,} |",
            f"| Total false negatives | {total_false_negatives:,} |",
            f"| Mean fold accuracy | {results['accuracy'].mean():.6f} |",
            f"| Mean fold attack recall | {results['attack_recall'].mean():.6f} |",
            f"| Mean fold benign recall | {results['benign_recall'].mean():.6f} |",
            f"| Mean fold macro F1 | {results['macro_f1'].mean():.6f} |",
            "",
            "## Per-Device Metrics",
            "",
            "| Held-Out Device | Device Name | Test Rows | Test Benign | Test Attack | Accuracy | Attack Precision | Attack Recall | Attack F1 | Benign Recall | Macro F1 | FP | FN | Train Seconds |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in results.sort_values("held_out_device_id").itertuples(index=False):
        lines.append(
            "| "
            f"{row.held_out_device_id} | "
            f"{row.held_out_device_name} | "
            f"{row.test_rows:,} | "
            f"{row.test_benign:,} | "
            f"{row.test_attack:,} | "
            f"{format_metric(row.accuracy)} | "
            f"{format_metric(row.attack_precision)} | "
            f"{format_metric(row.attack_recall)} | "
            f"{format_metric(row.attack_f1)} | "
            f"{format_metric(row.benign_recall)} | "
            f"{format_metric(row.macro_f1)} | "
            f"{row.true_benign_pred_attack:,} | "
            f"{row.true_attack_pred_benign:,} | "
            f"{row.train_seconds:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Device Confusion Matrices",
            "",
        ]
    )

    for row in results.sort_values("held_out_device_id").itertuples(index=False):
        lines.extend(
            [
                f"### Device {row.held_out_device_id}: {row.held_out_device_name}",
                "",
                "| Actual / Predicted | Benign | Attack |",
                "|---|---:|---:|",
                (
                    f"| benign | {row.true_benign_pred_benign:,} | "
                    f"{row.true_benign_pred_attack:,} |"
                ),
                (
                    f"| attack | {row.true_attack_pred_benign:,} | "
                    f"{row.true_attack_pred_attack:,} |"
                ),
                "",
            ]
        )

    worst_attack = results.sort_values(
        ["attack_recall", "true_attack_pred_benign"],
        ascending=[True, False],
    ).head(3)
    worst_benign = results.sort_values(
        ["benign_recall", "true_benign_pred_attack"],
        ascending=[True, False],
    ).head(3)

    lines.extend(
        [
            "## Hardest Devices",
            "",
            "### Lowest Attack Recall",
            "",
            "| Device | Device Name | Attack Recall | False Negatives | Test Attack Rows |",
            "|---:|---|---:|---:|---:|",
        ]
    )

    for row in worst_attack.itertuples(index=False):
        lines.append(
            "| "
            f"{row.held_out_device_id} | "
            f"{row.held_out_device_name} | "
            f"{format_metric(row.attack_recall)} | "
            f"{row.true_attack_pred_benign:,} | "
            f"{row.test_attack:,} |"
        )

    lines.extend(
        [
            "",
            "### Lowest Benign Recall",
            "",
            "| Device | Device Name | Benign Recall | False Positives | Test Benign Rows |",
            "|---:|---|---:|---:|---:|",
        ]
    )

    for row in worst_benign.itertuples(index=False):
        lines.append(
            "| "
            f"{row.held_out_device_id} | "
            f"{row.held_out_device_name} | "
            f"{format_metric(row.benign_recall)} | "
            f"{row.true_benign_pred_attack:,} | "
            f"{row.test_benign:,} |"
        )

    lines.extend(
        [
            "",
            "## Top20 Features Used",
            "",
            "| Rank | Feature |",
            "|---:|---|",
        ]
    )

    for index, feature in enumerate(feature_columns, start=1):
        lines.append(f"| {index} | `{feature}` |")

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "LODO should be interpreted differently from the stratified random split.",
            "",
            "- Stratified random split measures in-distribution performance when each device contributes rows to both train and test.",
            "- LODO measures cross-device generalization when the test device is completely unseen during training.",
            "- Lower LODO scores are expected and are more realistic for deployment to a new device type.",
            "",
            "For security detection, false negatives are the most important error type because they are attacks predicted as benign.",
            "",
            "## Output Files",
            "",
            f"- `{RESULTS_CSV_PATH}`",
            f"- `{CONFUSION_DIR}/device_<id>_confusion_matrix.csv`",
            f"- `{CONFUSION_DIR}/device_<id>_classification_report.txt`",
            "",
        ]
    )

    SUMMARY_REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    for path in [DEVICE_INFO_PATH, FEATURE_IMPORTANCE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

    device_info = load_device_info()
    device_ids = sorted(device_info)
    feature_columns = load_top_features()
    selected_columns = feature_columns + LABEL_COLUMNS

    device_paths = ensure_top20_device_csvs(device_ids, selected_columns)

    results = []
    for held_out_device_id in device_ids:
        results.append(
            evaluate_fold(
                held_out_device_id,
                device_info[held_out_device_id],
                device_paths,
                device_ids,
                feature_columns,
            )
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    write_markdown_report(results_df, device_info, feature_columns)

    print()
    print(f"Results CSV written to: {RESULTS_CSV_PATH}")
    print(f"Markdown report written to: {SUMMARY_REPORT_PATH}")


if __name__ == "__main__":
    main()
