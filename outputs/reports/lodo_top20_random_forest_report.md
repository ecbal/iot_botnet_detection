# LODO Top20 Random Forest Report

## Executive Summary

This report evaluates strict cross-device generalization with a Leave-One-Device-Out (LODO) setup.

Each fold holds out one IoT device completely as the test set and trains on the remaining eight devices. This is stricter than the earlier stratified random split because the held-out device contributes no rows to training.

The model uses the same top20 feature set selected from the all-device 115-feature Random Forest baseline.

## Experiment Setup

| Item | Value |
|---|---|
| Task | Binary classification: benign vs attack |
| Validation strategy | Leave-One-Device-Out |
| Number of folds | 9 |
| Model | RandomForestClassifier |
| Estimators | 100 |
| Random state | 42 |
| Features | 20 RF-importance-selected features |
| Scaling | none |
| Balancing | none |

## Devices

| Device ID | Device Name |
|---:|---|
| 1 | Danmini_Doorbell |
| 2 | Ecobee_Thermostat |
| 3 | Ennio_Doorbell |
| 4 | Philips_B120N10_Baby_Monitor |
| 5 | Provision_PT_737E_Security_Camera |
| 6 | Provision_PT_838_Security_Camera |
| 7 | Samsung_SNH_1011_N_Webcam |
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera |

## Overall LODO Result

| Metric | Value |
|---|---:|
| Total held-out test rows | 7,062,606 |
| Overall accuracy from pooled confusion counts | 0.999930 |
| Weighted attack recall from pooled counts | 0.999995 |
| Weighted benign recall from pooled counts | 0.999167 |
| Total false positives | 463 |
| Total false negatives | 31 |
| Mean fold accuracy | 0.999932 |
| Mean fold attack recall | 0.999996 |
| Mean fold benign recall | 0.998911 |
| Mean fold macro F1 | 0.999685 |

## Per-Device Metrics

| Held-Out Device | Device Name | Test Rows | Test Benign | Test Attack | Accuracy | Attack Precision | Attack Recall | Attack F1 | Benign Recall | Macro F1 | FP | FN | Train Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Danmini_Doorbell | 1,018,298 | 49,548 | 968,750 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 122.10 |
| 2 | Ecobee_Thermostat | 835,876 | 13,113 | 822,763 | 0.999999 | 0.999999 | 1.000000 | 0.999999 | 0.999924 | 0.999981 | 1 | 0 | 123.83 |
| 3 | Ennio_Doorbell | 355,500 | 39,100 | 316,400 | 0.999949 | 0.999943 | 1.000000 | 0.999972 | 0.999540 | 0.999871 | 18 | 0 | 129.66 |
| 4 | Philips_B120N10_Baby_Monitor | 1,098,677 | 175,240 | 923,437 | 0.999982 | 0.999978 | 1.000000 | 0.999989 | 0.999886 | 0.999966 | 20 | 0 | 114.84 |
| 5 | Provision_PT_737E_Security_Camera | 828,260 | 62,154 | 766,106 | 0.999976 | 1.000000 | 0.999974 | 0.999987 | 1.000000 | 0.999913 | 0 | 20 | 129.28 |
| 6 | Provision_PT_838_Security_Camera | 836,891 | 98,514 | 738,377 | 0.999994 | 0.999993 | 1.000000 | 0.999997 | 0.999949 | 0.999986 | 5 | 0 | 133.49 |
| 7 | Samsung_SNH_1011_N_Webcam | 375,222 | 52,150 | 323,072 | 0.999984 | 0.999981 | 1.000000 | 0.999991 | 0.999885 | 0.999967 | 6 | 0 | 147.42 |
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera | 863,056 | 46,585 | 816,471 | 0.999525 | 0.999499 | 0.999999 | 0.999749 | 0.991220 | 0.997665 | 409 | 1 | 138.88 |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera | 850,826 | 19,528 | 831,298 | 0.999984 | 0.999995 | 0.999988 | 0.999992 | 0.999795 | 0.999817 | 4 | 10 | 125.22 |

## Per-Device Confusion Matrices

### Device 1: Danmini_Doorbell

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 49,548 | 0 |
| attack | 0 | 968,750 |

### Device 2: Ecobee_Thermostat

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 13,112 | 1 |
| attack | 0 | 822,763 |

### Device 3: Ennio_Doorbell

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 39,082 | 18 |
| attack | 0 | 316,400 |

### Device 4: Philips_B120N10_Baby_Monitor

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 175,220 | 20 |
| attack | 0 | 923,437 |

### Device 5: Provision_PT_737E_Security_Camera

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 62,154 | 0 |
| attack | 20 | 766,086 |

### Device 6: Provision_PT_838_Security_Camera

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 98,509 | 5 |
| attack | 0 | 738,377 |

### Device 7: Samsung_SNH_1011_N_Webcam

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 52,144 | 6 |
| attack | 0 | 323,072 |

### Device 8: SimpleHome_XCS7_1002_WHT_Security_Camera

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 46,176 | 409 |
| attack | 1 | 816,470 |

### Device 9: SimpleHome_XCS7_1003_WHT_Security_Camera

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 19,524 | 4 |
| attack | 10 | 831,288 |

## Hardest Devices

### Lowest Attack Recall

| Device | Device Name | Attack Recall | False Negatives | Test Attack Rows |
|---:|---|---:|---:|---:|
| 5 | Provision_PT_737E_Security_Camera | 0.999974 | 20 | 766,106 |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera | 0.999988 | 10 | 831,298 |
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera | 0.999999 | 1 | 816,471 |

### Lowest Benign Recall

| Device | Device Name | Benign Recall | False Positives | Test Benign Rows |
|---:|---|---:|---:|---:|
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera | 0.991220 | 409 | 46,585 |
| 3 | Ennio_Doorbell | 0.999540 | 18 | 39,100 |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera | 0.999795 | 4 | 19,528 |

## Top20 Features Used

| Rank | Feature |
|---:|---|
| 1 | `HH_jit_L0.01_mean` |
| 2 | `HH_L0.01_radius` |
| 3 | `HpHp_L0.1_std` |
| 4 | `HpHp_L0.01_radius` |
| 5 | `HH_L0.01_covariance` |
| 6 | `HpHp_L0.1_radius` |
| 7 | `HpHp_L0.01_std` |
| 8 | `MI_dir_L0.01_weight` |
| 9 | `HH_L0.01_pcc` |
| 10 | `HpHp_L0.01_weight` |
| 11 | `HpHp_L0.1_weight` |
| 12 | `HH_L0.1_pcc` |
| 13 | `HH_jit_L0.1_variance` |
| 14 | `HpHp_L1_radius` |
| 15 | `HH_L0.1_covariance` |
| 16 | `MI_dir_L0.1_weight` |
| 17 | `HH_L0.1_radius` |
| 18 | `HH_jit_L5_mean` |
| 19 | `HH_L0.01_std` |
| 20 | `H_L1_weight` |

## Interpretation Guide

LODO should be interpreted differently from the stratified random split.

- Stratified random split measures in-distribution performance when each device contributes rows to both train and test.
- LODO measures cross-device generalization when the test device is completely unseen during training.
- Lower LODO scores are expected and are more realistic for deployment to a new device type.

For security detection, false negatives are the most important error type because they are attacks predicted as benign.

## Comparison Against Stratified Top20 RF

The earlier stratified top20 RF experiment used the same 20 features, but each device could contribute samples to both train and test.

| Experiment | Test Strategy | Test Rows | Accuracy | Attack Recall | Benign Recall | False Positives | False Negatives |
|---|---|---:|---:|---:|---:|---:|---:|
| Stratified top20 RF | Random stratified 80/20 | 1,412,521 | 0.999998 | 0.999999 | 0.999982 | 2 | 1 |
| LODO top20 RF | One held-out device per fold | 7,062,606 pooled across folds | 0.999930 | 0.999995 | 0.999167 | 463 | 31 |

The LODO result is still very strong, but it is meaningfully harder:

- False positives increased from 2 to 463 when evaluated across all held-out device folds.
- False negatives increased from 1 to 31.
- Attack recall remains extremely high at 0.999995 from pooled counts.
- Benign recall drops more noticeably than attack recall, mainly because Device 8 produced 409 benign false positives.

This is expected. LODO tests whether the model can generalize to an unseen device, while stratified random split mostly tests whether it can classify traffic similar to traffic already represented in training.

## Device-Level Interpretation

### Strongest Held-Out Device

Device 1, `Danmini_Doorbell`, was classified perfectly:

- 1,018,298 test rows
- 0 false positives
- 0 false negatives
- accuracy 1.000000

This means the top20 RF model trained on the other eight devices generalized cleanly to this device in the binary benign/attack setting.

### Main False Positive Driver

Device 8, `SimpleHome_XCS7_1002_WHT_Security_Camera`, is the main source of benign false positives:

- 409 benign rows predicted as attack
- benign recall 0.991220
- only 1 attack false negative

This indicates that the model is conservative on Device 8: it rarely misses attacks, but it is more likely to flag benign traffic as malicious for this device.

For a security-oriented system, this behavior may be acceptable if missing attacks is more costly than false alarms. For an operational monitoring system, this device would require closer analysis because false alarms can create alert fatigue.

### Main False Negative Drivers

The most missed attacks occur on:

| Device | Device Name | False Negatives |
|---:|---|---:|
| 5 | Provision_PT_737E_Security_Camera | 20 |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera | 10 |
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera | 1 |

Device 5 is the most important device to inspect if the project prioritizes reducing missed attacks. Even there, attack recall remains very high at 0.999974.

## Error Profile

Across all nine LODO folds:

```text
false positives: 463
false negatives: 31
total errors:    494
test rows:       7,062,606
```

The model makes many more false positives than false negatives in LODO:

```text
FP/FN ratio = 463 / 31 = 14.94
```

This suggests the model tends to err on the side of labeling uncertain traffic as attack rather than benign. In cybersecurity detection, this is usually preferable to missing attacks, but it still matters for alert management.

## Security Interpretation

From a security perspective, the most important number is attack recall:

```text
weighted attack recall = 0.999995
```

This means that, across all held-out device evaluations, only 31 attack samples out of 6,506,674 attack samples were classified as benign.

The practical interpretation:

- The model generalizes very well to unseen devices in the binary detection task.
- The remaining risk is concentrated in a small number of devices, especially Device 5 and Device 9 for false negatives.
- The larger operational issue is false positives, especially Device 8.

## Operational Interpretation

If this model were used as a detection component:

- It would catch almost all attacks across unseen devices.
- It would occasionally generate false alarms on benign traffic.
- Device 8 would need device-specific monitoring or threshold review because it contributes most benign false positives.
- The top20 feature set is compact enough to support a lighter feature extraction and model pipeline.

The result supports the idea that the top20 feature model is not only strong under random stratified evaluation but also robust under stricter cross-device evaluation.

## Presentation-Ready Key Messages

The following points can be used directly in a slide deck:

1. LODO was used as a stricter validation method to test cross-device generalization.
2. Each fold trained on eight IoT devices and tested on the one completely unseen device.
3. The same top20 feature set from RF feature importance was used for every fold.
4. Across all nine held-out device tests, the model achieved 0.999930 pooled accuracy.
5. Attack recall remained extremely high at 0.999995.
6. Only 31 attack samples were missed across 6,506,674 attack samples.
7. Most errors were false positives rather than false negatives.
8. Device 8 caused the majority of benign false positives.
9. Device 5 caused the most attack false negatives.
10. Compared with stratified random split, LODO is more realistic and harder because the test device is unseen during training.

## Suggested Slide Structure

### Slide 1: Why LODO?

Main message:

- Stratified random split gives an in-distribution baseline, but LODO tests unseen-device generalization.

Suggested content:

- Train on 8 devices
- Test on 1 held-out device
- Repeat for all 9 devices
- More realistic deployment scenario

### Slide 2: LODO Setup

Main message:

- The experiment uses the compact top20 feature model.

Suggested content:

- Random Forest, 100 estimators
- 20 selected features
- no SMOTE
- no scaling
- binary target: benign vs attack

### Slide 3: Overall LODO Results

Main message:

- Cross-device performance remains very high.

Suggested content:

- pooled accuracy: 0.999930
- weighted attack recall: 0.999995
- total false positives: 463
- total false negatives: 31

### Slide 4: Per-Device Results

Main message:

- Most devices generalize cleanly; Device 8 is the main false-positive outlier.

Suggested content:

- per-device accuracy table
- FP/FN by device
- highlight Device 1 perfect result
- highlight Device 8 false positives

### Slide 5: Error Analysis

Main message:

- Errors are mostly false alarms, not missed attacks.

Suggested content:

- FP: 463
- FN: 31
- false positives concentrated on Device 8
- false negatives concentrated on Device 5 and Device 9

### Slide 6: Stratified vs LODO

Main message:

- LODO is stricter and more realistic, so higher error counts are expected.

Suggested content:

- stratified top20: 3 total errors on 20% test set
- LODO top20: 494 total errors across all held-out device folds
- LODO still maintains 0.999995 attack recall

### Slide 7: Interpretation

Main message:

- The model generalizes well to unseen devices for binary attack detection.

Suggested content:

- top20 feature set is robust
- false negatives are rare
- false positives may need operational tuning

### Slide 8: Final Takeaway

Main message:

- Top20 RF is a strong lightweight model under both stratified and LODO evaluation.

Suggested content:

- good candidate for compact deployment
- strong attack recall
- additional work should target false positives on specific devices

## Recommended Next Analysis If Needed

No additional model family is required for the current scope. If more depth is needed, the most useful next analyses would be:

1. Inspect Device 8 benign false positives by `source_file`.
2. Inspect Device 5 and Device 9 attack false negatives by `source_file`.
3. Compare LODO top20 RF with LODO top20 + class weights instead of SMOTE.
4. Add device-level plots for FP/FN counts for presentation visuals.

These are analysis extensions, not blockers for the current report.

## Output Files

- `outputs/reports/lodo_top20_random_forest_results.csv`
- `outputs/reports/lodo_top20_confusion_matrices/device_<id>_confusion_matrix.csv`
- `outputs/reports/lodo_top20_confusion_matrices/device_<id>_classification_report.txt`
