# Publication Benchmark Report

## Run basis

One complete measured run per model was performed. Values are individual measurements, not averages; mean and standard deviation are therefore reported as null/—.

The dataset contains approximately seven million rows and each model worker reloads the full 115-feature canonical train/test splits before any timed Top-20 materialization. On the 16 GiB fanless MacBook Air, three complete repetitions would be disproportionately expensive and thermally confounded, so the prompt's accepted one-run protocol was used.

## System information

| Item | Value |
| --- | ---: |
| Operating system | macOS 26.5.2 |
| Kernel | 25.5.0 |
| Architecture | arm64 |
| Computer | MacBook Air |
| CPU/chip | Apple M4 |
| Physical/logical cores | 10/10 |
| RAM | 16.00 GiB |
| Python | 3.14.3 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| NumPy | 2.4.4 |
| imbalanced-learn | 0.14.1 |
| psutil | 7.2.2 |
| joblib | 1.5.3 |

## Dataset and split

The exact saved standard split files were used for every model. The split is global, row-level, and stratified by `binary_target`; it is not a temporal or unseen-device split.

| Split/stage | Rows | Benign | Benign (%) | Attack | Attack (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train before SMOTE | 5650085 | 444746 | 7.871492 | 5205339 | 92.128508 |
| Test (never resampled) | 1412521 | 111186 | 7.871458 | 1301335 | 92.128542 |
| SMOTE train after resampling | 10410678 | 5205339 | 50.000000 | 5205339 | 50.000000 |

## Model and preprocessing methodology

- Random Forest: `{'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'verbose': 1}`. The existing scripts explicitly include `verbose=1`; all other parameters use scikit-learn defaults.
- RF Baseline uses all 115 features with no scaling or balancing.
- RF Top-20 uses the existing fixed Top-20 set with no scaling or balancing.
- Top-20 + SMOTE uses default `StandardScaler` fitted on train only, transforms train and test separately, then applies `SMOTE(random_state=42, k_neighbors=5)` to training data only.
- Attack (`binary_target=1`) is the positive class. Metrics use `average='binary'`, `pos_label=1`, and `zero_division=0`.

### Fixed Top-20 provenance and input order

The same fixed set is used by both Top-20 configurations. Importance rank is
recorded separately from model input order because the existing cached CSV/model
pipeline preserves canonical source-column order; column permutation can change a
fixed-seed Random Forest's fitted trees.

| Importance rank | Ranked feature |
| --- | ---: |
| 1 | HH_jit_L0.01_mean |
| 2 | HH_L0.01_radius |
| 3 | HpHp_L0.1_std |
| 4 | HpHp_L0.01_radius |
| 5 | HH_L0.01_covariance |
| 6 | HpHp_L0.1_radius |
| 7 | HpHp_L0.01_std |
| 8 | MI_dir_L0.01_weight |
| 9 | HH_L0.01_pcc |
| 10 | HpHp_L0.01_weight |
| 11 | HpHp_L0.1_weight |
| 12 | HH_L0.1_pcc |
| 13 | HH_jit_L0.1_variance |
| 14 | HpHp_L1_radius |
| 15 | HH_L0.1_covariance |
| 16 | MI_dir_L0.1_weight |
| 17 | HH_L0.1_radius |
| 18 | HH_jit_L5_mean |
| 19 | HH_L0.01_std |
| 20 | H_L1_weight |

| Model column position | Model input feature |
| --- | ---: |
| 1 | MI_dir_L0.1_weight |
| 2 | MI_dir_L0.01_weight |
| 3 | H_L1_weight |
| 4 | HH_L0.1_radius |
| 5 | HH_L0.1_covariance |
| 6 | HH_L0.1_pcc |
| 7 | HH_L0.01_std |
| 8 | HH_L0.01_radius |
| 9 | HH_L0.01_covariance |
| 10 | HH_L0.01_pcc |
| 11 | HH_jit_L5_mean |
| 12 | HH_jit_L0.1_variance |
| 13 | HH_jit_L0.01_mean |
| 14 | HpHp_L1_radius |
| 15 | HpHp_L0.1_weight |
| 16 | HpHp_L0.1_std |
| 17 | HpHp_L0.1_radius |
| 18 | HpHp_L0.01_weight |
| 19 | HpHp_L0.01_std |
| 20 | HpHp_L0.01_radius |

## Timing definitions

- **Feature-selection time:** `null` for Top-20 models because the fixed list was precomputed by the existing baseline and was not selected again during this benchmark. `feature_selection_precomputed=true`.
- **Data-loading time:** CSV parsing plus materialization of the full 115-feature matrix and target, measured separately for train and test. It is excluded from preprocessing.
- **Feature-subsetting time:** actual `.copy(deep=True)` materialization of the selected columns from each newly loaded 115-feature matrix, measured separately for train and test.
- **Scaling time:** scaler fit, train transform, and test transform are separate raw measurements.
- **SMOTE time:** only `SMOTE.fit_resample()` on the scaled training data.
- **Training time:** only `RandomForestClassifier.fit()`.
- **Prediction time:** only `RandomForestClassifier.predict()` on the test split. The
  Top-20 + SMOTE configuration applies its train-fitted scaler to test features, as in
  the existing method, but never resamples the test rows.
- **Preprocessing:** feature subsetting + scaling + SMOTE, as applicable. Data loading, fitting, and prediction are excluded.
- **Total pipeline:** preprocessing + training + prediction; the primary table excludes I/O.

| Experiment | Feature selection (s) | Train subset (s) | Test subset (s) | Scaler fit (s) | Train scale (s) | Test scale (s) | SMOTE (s) | RF fit (s) | Predict (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF Baseline | — | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 256.461498 | 0.683392 |
| RF Top-20 | — | 0.338378 | 0.068085 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 104.741095 | 0.404588 |
| Top-20 + SMOTE | — | 0.340737 | 0.073678 | 0.352703 | 0.048442 | 0.030384 | 103.662409 | 246.318821 | 0.277028 |

### I/O and full-pipeline timing

| Experiment | Train load (s) | Test load (s) | Total load (s) | Pipeline excluding I/O (s) | Pipeline including I/O (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF Baseline | 24.989449 | 6.339257 | 31.328706 | 257.144889 | 288.473595 |
| RF Top-20 | 25.299784 | 6.256529 | 31.556313 | 105.552146 | 137.108459 |
| Top-20 + SMOTE | 25.041615 | 6.248718 | 31.290333 | 351.104202 | 382.394535 |

## Runtime results

| Experiment | Feature Count | Feature Subsetting (s) | Scaling (s) | SMOTE (s) | Preprocessing (s) | Training (s) | Prediction (s) | Total Pipeline (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF Baseline | 115 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 256.461498 | 0.683392 | 257.144889 |
| RF Top-20 | 20 | 0.406463 | 0.000000 | 0.000000 | 0.406463 | 104.741095 | 0.404588 | 105.552146 |
| Top-20 + SMOTE | 20 | 0.414415 | 0.431529 | 103.662409 | 104.508353 | 246.318821 | 0.277028 | 351.104202 |

## Dataset, RSS, and model-size results

Logical dataset memory and process RSS are distinct. MB values use binary conversion (`bytes / 1,048,576`). Peak RSS samples the isolated worker PID, including all RF threads, every 0.05 seconds. Serialized model size includes only the Random Forest estimator; the SMOTE scaler and feature metadata are excluded to preserve existing project practice.

| Experiment | Training Dataset Memory (MB) | Test Dataset Memory (MB) | Peak Training RSS (MB) | Incremental Peak Training RSS (MB) | Model Size (MB) | Compressed Model Size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF Baseline | 2484.025430 | 621.006436 | 4542.500000 | 3007.890625 | 2.034799 | 0.605749 |
| RF Top-20 | 436.455869 | 109.114137 | 4424.500000 | 3930.250000 | 2.462152 | 0.705954 |
| Top-20 + SMOTE | 804.200220 | 109.114011 | 5710.906250 | 1797.828125 | 2.635522 | 0.789251 |

## Classification performance

| Experiment | Accuracy | Precision | Recall | F1-score | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF Baseline | 0.999999292 | 0.999999232 | 1.000000000 | 0.999999616 | 111185 | 1 | 0 | 1301335 |
| RF Top-20 | 0.999997876 | 0.999998463 | 0.999999232 | 0.999998847 | 111184 | 2 | 1 | 1301334 |
| Top-20 + SMOTE | 0.999998584 | 0.999999232 | 0.999999232 | 0.999999232 | 111185 | 1 | 1 | 1301334 |

## RF Top-20 versus RF Baseline

- Feature count: 82.61% reduction (115.000000 → 20.000000).
- Training time: 59.16% reduction (256.461498 s → 104.741095 s).
- Prediction time: 40.80% reduction (0.683392 s → 0.404588 s).
- Training dataset memory: 82.43% reduction (2484.025430 MB → 436.455869 MB).
- Test dataset memory: 82.43% reduction (621.006436 MB → 109.114137 MB).
- Peak training RSS: 2.60% reduction (4542.500000 MB → 4424.500000 MB).
- Incremental training RSS: 30.66% increase (3007.890625 MB → 3930.250000 MB).
- Total pipeline excluding I/O: 58.95% reduction (257.144889 s → 105.552146 s).
- Uncompressed model size: 21.00% increase (2.034799 MB → 2.462152 MB).
- Compressed model size: 16.54% increase (0.605749 MB → 0.705954 MB).

- Training ratio: 2.4485× (speedup).
- Additional preprocessing relative to baseline: 0.406463 s.
- Accuracy difference: -0.000141591 percentage points.
- Precision difference: -0.000076844 percentage points.
- Recall difference: -0.000076844 percentage points.
- F1 difference: -0.000076844 percentage points.

## Top-20 + SMOTE overhead versus RF Top-20

- Additional preprocessing time: 104.101889 s.
- SMOTE-only runtime: 103.662409 s.
- Training rows: 84.26% increase (5650085.000000 → 10410678.000000).
- Training dataset memory: 84.26% increase (436.455869 MB → 804.200220 MB).
- Training time: 135.17% increase (104.741095 s → 246.318821 s).
- Total pipeline excluding I/O: 232.64% increase (105.552146 s → 351.104202 s).
- Peak training RSS: 29.07% increase (4424.500000 MB → 5710.906250 MB).

- Accuracy difference: 0.000070795 percentage points.
- Precision difference: 0.000076844 percentage points.
- Recall difference: 0.000000000 percentage points.
- F1 difference: 0.000038422 percentage points.

## Existing-output cross-check

| Experiment | Confusion matrix matches existing | Existing report train (s) | New measured train (s) | Existing report predict (s) | New measured predict (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF Baseline | True | 418.560000 | 256.461498 | 1.050000 | 0.683392 |
| RF Top-20 | True | 82.310000 | 104.741095 | 0.420000 | 0.404588 |
| Top-20 + SMOTE | True | 260.800000 | 246.318821 | 0.350000 | 0.277028 |

Runtime differences from earlier reports are expected because those are separate single executions with different cache, load, memory-pressure, and thermal states. A confusion-matrix mismatch, if present, is reported rather than hidden; this benchmark materializes Top-20 directly from the canonical 115-feature CSV values, whereas the existing reduced CSVs underwent an additional CSV parse/write round trip.

### Prior resource-benchmark cross-check

| Experiment | Prior resource fit (s) | New fit (s) | Prior resource predict (s) | New predict (s) | Prior peak fit RSS (MB) | New peak fit RSS (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF Baseline | 329.611340 | 256.461498 | 0.864880 | 0.683392 | 5321.656250 | 4542.500000 |
| RF Top-20 | 118.199102 | 104.741095 | 0.602719 | 0.404588 | 4230.703125 | 4424.500000 |
| Top-20 + SMOTE | 244.034590 | 246.318821 | 0.334982 | 0.277028 | 5292.718750 | 5710.906250 |

The earlier resource benchmark loaded already reduced Top-20 CSVs and therefore did
not measure the required 115→20 materialization; it also combined scaling phases.
Its runtime/RSS scope is not directly interchangeable with this publication benchmark.
The old standalone baseline raw-run JSON remains malformed, although its aggregate
comparison JSON used above is valid. Classification results and all six newly saved
model files match the existing artifacts exactly.

## Automatic validation

| Validation | Passed |
| --- | ---: |
| all_requested_runs_completed | True |
| feature_counts_are_115_20_20 | True |
| top20_models_use_identical_features_and_order | True |
| all_models_use_same_test_source | True |
| all_models_have_identical_test_target_fingerprint | True |
| all_models_have_same_test_rows_and_counts | True |
| smote_never_changed_test_target | True |
| all_worker_required_checks_passed | True |
| all_byte_to_mb_conversions_correct | True |
| no_preexisting_file_modified_or_created_outside_output_root | True |
| comparison_calculations_are_mathematically_correct | True |

## Limitations

- One complete measured run per model was used because each worker reloads roughly seven million rows, the full benchmark contains three large RF fits plus SMOTE, and the 16 GiB fanless machine makes three full repetitions impractical. Values are individual measurements; mean and standard deviation are null.
- Runtime and RSS are machine-state-dependent. The benchmark records initial resource state but does not control the macOS filesystem cache, background processes, power policy, or thermal throttling.
- The standard saved split is globally row-stratified; devices and source files occur on both sides. These results measure in-distribution performance, not unseen-device generalization.
- The saved CSVs do not contain stable original-row identifiers. The benchmark verifies distinct immutable train/test files and dataflow isolation, but cannot prove semantic row disjointness by comparing feature values because legitimate duplicates may exist.
- Top-20 matrices are deliberately materialized from the canonical 115-feature CSVs as required. Existing reduced CSVs underwent an additional CSV round trip with negligible floating-point last-bit differences, so a fresh confusion matrix can differ.
- Serialized size covers only RandomForestClassifier, matching existing project practice; the SMOTE model's fitted scaler and feature metadata are not bundled in model size.
- Feature-selection runtime is null: the fixed Top-20 list is a precomputed artifact and was intentionally not selected again in this fixed-list benchmark.
- The during-run no-overwrite manifest compared size and mtime_ns for all 246 pre-existing workspace files, not a full before/after content hash. The five canonical input artifacts were SHA-256 hashed after measurement and match their audited values.
- The initial run did not capture the exact executing benchmark-script SHA-256. Post-run changes affected schema naming, validation, OS labeling, and report formatting only; raw measurements and model files were not rerun or altered.

## Simplified runtime table for the paper

| Experiment | Preprocessing (s) | Training (s) | Prediction (s) |
| --- | ---: | ---: | ---: |
| RF Baseline | 0.000000 | 256.461498 | 0.683392 |
| RF Top-20 | 0.406463 | 104.741095 | 0.404588 |
| Top-20 + SMOTE | 104.508353 | 246.318821 | 0.277028 |

## Simplified resource table for the paper

| Experiment | Train Data Memory (MB) | Peak Training RAM (MB) | Model Size (MB) |
| --- | ---: | ---: | ---: |
| RF Baseline | 2484.025430 | 4542.500000 | 2.034799 |
| RF Top-20 | 436.455869 | 4424.500000 | 2.462152 |
| Top-20 + SMOTE | 804.200220 | 5710.906250 | 2.635522 |
