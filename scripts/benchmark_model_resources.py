from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


REPORT_DIR = Path("outputs/reports")
MODEL_DIR = Path("outputs/models")
TMP_DIR = Path("outputs/reports/model_resource_runs")
CSV_PATH = REPORT_DIR / "model_resource_comparison.csv"
JSON_PATH = REPORT_DIR / "model_resource_comparison.json"

TRAIN_115_PATH = Path("data/splits/all_devices_train_stratified.csv")
TEST_115_PATH = Path("data/splits/all_devices_test_stratified.csv")
TRAIN_TOP20_PATH = Path("data/splits/all_devices_train_stratified_top20.csv")
TEST_TOP20_PATH = Path("data/splits/all_devices_test_stratified_top20.csv")

TARGET_COLUMN = "binary_target"
DROP_COLUMNS = {"binary_label", "binary_target", "source_file"}
RANDOM_STATE = 42
MODEL_CONFIG = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 1,
}

MODEL_SPECS = {
    "rf_baseline_115": {
        "label": "RF Baseline using all 115 features",
        "evaluation_setup": "all_devices_stratified",
        "train_path": TRAIN_115_PATH,
        "test_path": TEST_115_PATH,
        "model_path": MODEL_DIR / "rf_baseline_115.joblib",
        "compressed_model_path": MODEL_DIR / "rf_baseline_115_compress3.joblib",
        "use_scaling": False,
        "use_smote": False,
    },
    "rf_top20": {
        "label": "RF using selected Top-20 features",
        "evaluation_setup": "all_devices_stratified_top20",
        "train_path": TRAIN_TOP20_PATH,
        "test_path": TEST_TOP20_PATH,
        "model_path": MODEL_DIR / "rf_top20.joblib",
        "compressed_model_path": MODEL_DIR / "rf_top20_compress3.joblib",
        "use_scaling": False,
        "use_smote": False,
    },
    "rf_top20_smote": {
        "label": "RF using selected Top-20 features with train-only scaling and SMOTE",
        "evaluation_setup": "all_devices_stratified_top20_smote",
        "train_path": TRAIN_TOP20_PATH,
        "test_path": TEST_TOP20_PATH,
        "model_path": MODEL_DIR / "rf_top20_smote.joblib",
        "compressed_model_path": MODEL_DIR / "rf_top20_smote_compress3.joblib",
        "use_scaling": True,
        "use_smote": True,
    },
}

META_COLUMNS = DROP_COLUMNS | {"device", "device_id", "device_identifier"}


def get_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "operating_system": platform.system(),
        "operating_system_version": platform.mac_ver()[0]
        if platform.system() == "Darwin"
        else platform.release(),
        "kernel_version": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "total_ram_bytes": psutil.virtual_memory().total,
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
    }

    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                check=True,
                capture_output=True,
                text=True,
            )
            hardware = json.loads(completed.stdout)["SPHardwareDataType"][0]
            info.update(
                {
                    "computer_model": hardware.get("machine_name"),
                    "model_identifier": hardware.get("machine_model"),
                    "chip": hardware.get("chip_type"),
                }
            )
        except (OSError, subprocess.SubprocessError, KeyError, IndexError, json.JSONDecodeError):
            pass

    return info


@dataclass
class MemorySample:
    before_rss_bytes: int
    peak_rss_bytes: int
    after_rss_bytes: int
    seconds: float


class RssSampler:
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.parent = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = self.total_rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def total_rss_bytes(self) -> int:
        total = 0
        processes = [self.parent]
        try:
            processes.extend(self.parent.children(recursive=True))
        except (psutil.Error, OSError):
            pass

        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.Error, OSError):
                continue
        return total

    def __enter__(self) -> "RssSampler":
        self.peak_rss_bytes = self.total_rss_bytes()
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, self.total_rss_bytes())

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, self.total_rss_bytes())
            self._stop.wait(self.interval_seconds)


def bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024 * 1024)


def get_feature_columns(path: Path) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    feature_columns = [column for column in columns if column not in DROP_COLUMNS]
    accidental_metadata = sorted(set(feature_columns) & META_COLUMNS)
    if accidental_metadata:
        raise ValueError(f"Metadata columns included as features: {accidental_metadata}")
    return feature_columns


def read_features_target(path: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    dtype = {column: "float32" for column in feature_columns}
    dtype[TARGET_COLUMN] = "uint8"
    frame = pd.read_csv(path, usecols=[*feature_columns, TARGET_COLUMN], dtype=dtype)
    y = frame.pop(TARGET_COLUMN)
    return frame, y


def frame_memory_bytes(frame: pd.DataFrame) -> int:
    return int(frame.memory_usage(deep=True).sum())


def series_memory_bytes(series: pd.Series) -> int:
    return int(series.memory_usage(deep=True))


def array_memory_bytes(array: Any) -> int:
    if hasattr(array, "memory_usage"):
        memory = array.memory_usage(deep=True)
        return int(memory.sum() if hasattr(memory, "sum") else memory)
    return int(np.asarray(array).nbytes)


def run_timed_with_memory(function: Any) -> tuple[Any, MemorySample]:
    before = RssSampler().total_rss_bytes()
    start = perf_counter()
    with RssSampler(interval_seconds=0.1) as sampler:
        result = function()
    seconds = perf_counter() - start
    after = sampler.total_rss_bytes()
    return result, MemorySample(
        before_rss_bytes=before,
        peak_rss_bytes=sampler.peak_rss_bytes,
        after_rss_bytes=after,
        seconds=seconds,
    )


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[RandomForestClassifier, MemorySample]:
    model = RandomForestClassifier(**MODEL_CONFIG)
    fitted_model, sample = run_timed_with_memory(lambda: model.fit(X_train, y_train))
    return fitted_model, sample


def predict_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
) -> tuple[pd.Series, MemorySample]:
    return run_timed_with_memory(lambda: model.predict(X_test))


def save_model_sizes(model: RandomForestClassifier, spec: dict[str, Any]) -> dict[str, float | int]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = spec["model_path"]
    compressed_path = spec["compressed_model_path"]

    joblib.dump(model, model_path, compress=0)
    joblib.dump(model, compressed_path, compress=3)

    model_size_bytes = model_path.stat().st_size
    compressed_size_bytes = compressed_path.stat().st_size
    return {
        "model_path": str(model_path),
        "compressed_model_path": str(compressed_path),
        "model_size_bytes": model_size_bytes,
        "model_size_mb": bytes_to_mb(model_size_bytes),
        "compressed_model_size_bytes": compressed_size_bytes,
        "compressed_model_size_mb": bytes_to_mb(compressed_size_bytes),
    }


def run_child(model_key: str, run_index: int, output_path: Path) -> None:
    spec = MODEL_SPECS[model_key]
    feature_columns = get_feature_columns(spec["train_path"])
    X_train, y_train = read_features_target(spec["train_path"], feature_columns)
    X_test, y_test = read_features_target(spec["test_path"], feature_columns)

    train_rows_before_smote = len(X_train)
    train_rows_after_smote = len(X_train)
    train_dataset_memory_bytes = frame_memory_bytes(X_train) + series_memory_bytes(y_train)
    test_dataset_memory_bytes = frame_memory_bytes(X_test) + series_memory_bytes(y_test)
    y_train_memory_bytes = series_memory_bytes(y_train)
    X_train_memory_bytes = frame_memory_bytes(X_train)
    X_test_memory_bytes = frame_memory_bytes(X_test)

    scaling_time_seconds = 0.0
    smote_time_seconds = 0.0
    resampled_train_memory_bytes = 0

    if spec["use_scaling"]:
        scaler = StandardScaler()
        scale_start = perf_counter()
        X_train_for_fit = scaler.fit_transform(X_train).astype("float32", copy=False)
        X_test_for_predict = scaler.transform(X_test).astype("float32", copy=False)
        scaling_time_seconds = perf_counter() - scale_start
    else:
        X_train_for_fit = X_train
        X_test_for_predict = X_test

    if spec["use_smote"]:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        smote_start = perf_counter()
        X_train_for_fit, y_train_for_fit = smote.fit_resample(X_train_for_fit, y_train)
        smote_time_seconds = perf_counter() - smote_start
        train_rows_after_smote = len(y_train_for_fit)
        resampled_train_memory_bytes = int(
            array_memory_bytes(X_train_for_fit) + array_memory_bytes(y_train_for_fit)
        )
    else:
        y_train_for_fit = y_train
        if hasattr(X_train_for_fit, "memory_usage"):
            resampled_train_memory_bytes = frame_memory_bytes(X_train_for_fit) + series_memory_bytes(y_train_for_fit)
        else:
            resampled_train_memory_bytes = int(
                array_memory_bytes(X_train_for_fit) + array_memory_bytes(y_train_for_fit)
            )

    model, fit_sample = fit_model(X_train_for_fit, y_train_for_fit)
    y_pred, predict_sample = predict_model(model, X_test_for_predict)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    model_sizes = save_model_sizes(model, spec)

    result = {
        "model": model_key,
        "model_label": spec["label"],
        "run_index": run_index,
        "evaluation_setup": spec["evaluation_setup"],
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "train_rows_before_smote": train_rows_before_smote,
        "train_rows_after_smote": train_rows_after_smote,
        "test_rows": len(X_test),
        "X_train_memory_bytes": X_train_memory_bytes,
        "X_train_memory_mb": bytes_to_mb(X_train_memory_bytes),
        "X_test_memory_bytes": X_test_memory_bytes,
        "X_test_memory_mb": bytes_to_mb(X_test_memory_bytes),
        "y_train_memory_bytes": y_train_memory_bytes,
        "y_train_memory_mb": bytes_to_mb(y_train_memory_bytes),
        "train_dataset_memory_bytes": train_dataset_memory_bytes,
        "train_dataset_memory_mb": bytes_to_mb(train_dataset_memory_bytes),
        "test_dataset_memory_bytes": test_dataset_memory_bytes,
        "test_dataset_memory_mb": bytes_to_mb(test_dataset_memory_bytes),
        "resampled_train_memory_bytes": resampled_train_memory_bytes,
        "resampled_train_memory_mb": bytes_to_mb(resampled_train_memory_bytes),
        "pre_fit_rss_bytes": fit_sample.before_rss_bytes,
        "pre_fit_rss_mb": bytes_to_mb(fit_sample.before_rss_bytes),
        "peak_fit_rss_bytes": fit_sample.peak_rss_bytes,
        "peak_fit_rss_mb": bytes_to_mb(fit_sample.peak_rss_bytes),
        "incremental_peak_fit_rss_bytes": fit_sample.peak_rss_bytes - fit_sample.before_rss_bytes,
        "incremental_peak_fit_rss_mb": bytes_to_mb(fit_sample.peak_rss_bytes - fit_sample.before_rss_bytes),
        "post_fit_rss_bytes": fit_sample.after_rss_bytes,
        "post_fit_rss_mb": bytes_to_mb(fit_sample.after_rss_bytes),
        "pre_predict_rss_bytes": predict_sample.before_rss_bytes,
        "pre_predict_rss_mb": bytes_to_mb(predict_sample.before_rss_bytes),
        "peak_predict_rss_bytes": predict_sample.peak_rss_bytes,
        "peak_predict_rss_mb": bytes_to_mb(predict_sample.peak_rss_bytes),
        "incremental_peak_predict_rss_bytes": predict_sample.peak_rss_bytes
        - predict_sample.before_rss_bytes,
        "incremental_peak_predict_rss_mb": bytes_to_mb(
            predict_sample.peak_rss_bytes - predict_sample.before_rss_bytes
        ),
        "post_predict_rss_bytes": predict_sample.after_rss_bytes,
        "post_predict_rss_mb": bytes_to_mb(predict_sample.after_rss_bytes),
        "scaling_time_seconds": scaling_time_seconds,
        "smote_time_seconds": smote_time_seconds,
        "training_time_seconds": fit_sample.seconds,
        "prediction_time_seconds": predict_sample.seconds,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        **model_sizes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    del model, X_train, X_test, y_train, y_test, X_train_for_fit, X_test_for_predict, y_train_for_fit, y_pred
    gc.collect()


def numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {"run_index"}
    columns = []
    for key, value in rows[0].items():
        if key in excluded:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            columns.append(key)
    return columns


def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": rows[0]["model"],
        "model_label": rows[0]["model_label"],
        "evaluation_setup": rows[0]["evaluation_setup"],
        "runs": len(rows),
        "feature_count": rows[0]["feature_count"],
        "train_rows_before_smote": rows[0]["train_rows_before_smote"],
        "train_rows_after_smote": rows[0]["train_rows_after_smote"],
        "test_rows": rows[0]["test_rows"],
        "model_path": rows[-1]["model_path"],
        "compressed_model_path": rows[-1]["compressed_model_path"],
    }

    for column in numeric_columns(rows):
        values = [float(row[column]) for row in rows]
        summary[column] = mean(values)
        summary[f"{column}_min"] = min(values)
        summary[f"{column}_max"] = max(values)
        summary[f"{column}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return summary


def reduction_percent(baseline: float, compact: float) -> float:
    return ((baseline - compact) / baseline) * 100 if baseline else 0.0


def compute_reductions(summaries: list[dict[str, Any]]) -> dict[str, float]:
    by_model = {row["model"]: row for row in summaries}
    baseline = by_model["rf_baseline_115"]
    top20 = by_model["rf_top20"]
    return {
        "feature_count_reduction_percent": reduction_percent(
            baseline["feature_count"], top20["feature_count"]
        ),
        "training_time_reduction_percent": reduction_percent(
            baseline["training_time_seconds"], top20["training_time_seconds"]
        ),
        "peak_training_ram_reduction_percent": reduction_percent(
            baseline["peak_fit_rss_mb"], top20["peak_fit_rss_mb"]
        ),
        "training_dataset_memory_reduction_percent": reduction_percent(
            baseline["train_dataset_memory_mb"], top20["train_dataset_memory_mb"]
        ),
        "serialized_model_size_reduction_percent": reduction_percent(
            baseline["model_size_mb"], top20["model_size_mb"]
        ),
        "prediction_time_reduction_percent": reduction_percent(
            baseline["prediction_time_seconds"], top20["prediction_time_seconds"]
        ),
        "training_speedup": baseline["training_time_seconds"] / top20["training_time_seconds"],
    }


def write_reports(run_rows: list[dict[str, Any]], requested_runs: int) -> None:
    summaries = [summarize_model([row for row in run_rows if row["model"] == model]) for model in MODEL_SPECS]
    reductions = compute_reductions(summaries)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(CSV_PATH, index=False)
    JSON_PATH.write_text(
        json.dumps(
            {
                "requested_runs_per_model": requested_runs,
                "completed_runs_per_model": {
                    model: len([row for row in run_rows if row["model"] == model])
                    for model in MODEL_SPECS
                },
                "limitation": (
                    "One complete run per model was used because the all-devices dataset has "
                    "approximately seven million rows and the existing full RF baseline takes "
                    "several minutes per run."
                    if requested_runs == 1
                    else None
                ),
                "system_info": get_system_info(),
                "model_config": MODEL_CONFIG,
                "raw_runs": run_rows,
                "summary": summaries,
                "top20_vs_baseline_reductions": reductions,
            },
            indent=2,
        )
    )

    display_columns = [
        "model",
        "feature_count",
        "train_rows_before_smote",
        "train_rows_after_smote",
        "train_dataset_memory_mb",
        "peak_fit_rss_mb",
        "incremental_peak_fit_rss_mb",
        "training_time_seconds",
        "prediction_time_seconds",
        "model_size_mb",
        "compressed_model_size_mb",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]
    print()
    print("Concise comparison table")
    print(pd.DataFrame(summaries)[display_columns].to_string(index=False))
    print()
    print("Top-20 reductions vs 115-feature baseline")
    for key, value in reductions.items():
        suffix = "x" if key == "training_speedup" else "%"
        print(f"{key}: {value:.2f}{suffix}")
    print()
    print(f"CSV written to: {CSV_PATH}")
    print(f"JSON written to: {JSON_PATH}")


def validate_existing_run(row: dict[str, Any], model_key: str, run_index: int) -> None:
    expected_features = get_feature_columns(MODEL_SPECS[model_key]["train_path"])
    if row.get("model") != model_key or row.get("run_index") != run_index:
        raise ValueError(f"Existing run does not match {model_key} run {run_index}")
    if row.get("feature_columns") != expected_features:
        raise ValueError(f"Existing run has a different feature list for {model_key}")


def run_parent(runs: int, resume: bool) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    run_rows = []
    for model_key in MODEL_SPECS:
        for run_index in range(1, runs + 1):
            output_path = TMP_DIR / f"{model_key}_run_{run_index}.json"
            if resume and output_path.exists():
                row = json.loads(output_path.read_text())
                validate_existing_run(row, model_key, run_index)
                print(f"Reusing verified {model_key} run {run_index}/{runs}...")
                run_rows.append(row)
                continue
            cmd = [
                sys.executable,
                str(Path(__file__)),
                "--child",
                "--model",
                model_key,
                "--run-index",
                str(run_index),
                "--output",
                str(output_path),
            ]
            print(f"Running {model_key} run {run_index}/{runs}...")
            subprocess.run(cmd, check=True)
            run_rows.append(json.loads(output_path.read_text()))
            gc.collect()
    write_reports(run_rows, runs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing raw run JSON only after validating its model, run index, and feature list.",
    )
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.child:
        if args.model is None or args.output is None:
            raise ValueError("--child requires --model and --output")
        run_child(args.model, args.run_index, args.output)
    else:
        if args.runs < 1:
            raise ValueError("--runs must be at least 1")
        run_parent(args.runs, args.resume)


if __name__ == "__main__":
    main()
