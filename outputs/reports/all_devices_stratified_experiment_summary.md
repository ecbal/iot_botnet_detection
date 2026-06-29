# All Devices Stratified Binary Classification Experiment Report

## Executive Summary

This report summarizes the binary classification experiments performed on the full N-BaIoT dataset after labeling and combining all available device CSV files.

The goal of this experiment group was to detect whether network traffic is `benign` or `attack` using the 115 statistical network traffic features provided by the dataset. All experiments used a stratified 80/20 train/test split over the combined all-device labeled dataset.

Three Random Forest based experiments were completed:

1. Random Forest baseline using all 115 features.
2. Random Forest using the top 20 features selected from baseline RF feature importance.
3. Random Forest using the same top 20 features with SMOTE applied only to the training set.

The strongest raw detection result came from the 115-feature Random Forest baseline. It produced zero missed attacks and only one benign false alarm on the test set.

The top20 Random Forest model achieved nearly identical performance while reducing training time from 418.56 seconds to 82.31 seconds. This makes it a strong lightweight candidate if model simplicity, speed, or feature reduction matters.

The top20 + SMOTE experiment did not materially improve attack detection. It reduced benign false alarms from two to one compared with top20 RF, but still missed one attack and required a larger synthetic training set.

## High-Level Takeaway

| Recommendation Area | Best Current Choice | Reason |
|---|---|---|
| Best raw detection | 115-feature RF baseline | No missed attacks, only one false alarm |
| Best speed/performance tradeoff | Top20 RF | About 5.1x faster training with only three total test errors |
| Best balancing experiment | Top20 + SMOTE RF | Balanced training data, but no clear detection gain |
| Current modeling direction | Keep RF baseline and top20 RF | SMOTE is not clearly justified for this binary setup |

## Dataset Preparation

### Source Dataset

The experiments use the N-BaIoT dataset stored under:

```text
archive-2/
```

The raw dataset contains network traffic CSV files split by device and traffic type. File names encode both the device id and the traffic class.

Examples:

```text
1.benign.csv
1.gafgyt.combo.csv
1.mirai.udp.csv
2.benign.csv
...
9.mirai.udpplain.csv
```

### Device Coverage

All nine available device ids were included:

```text
device_1
device_2
device_3
device_4
device_5
device_6
device_7
device_8
device_9
```

Some devices do not include every Mirai attack subtype in the raw dataset. The labeling process uses whatever raw CSV files exist for each device.

### Binary Labeling Rule

The binary label was derived from each source file name:

| Source file pattern | `binary_label` | `binary_target` |
|---|---|---:|
| `*.benign.csv` | `benign` | 0 |
| Any Gafgyt/BASHLITE file | `attack` | 1 |
| Any Mirai file | `attack` | 1 |

Each labeled row includes:

- the original 115 network traffic features
- `binary_label`
- `binary_target`
- `source_file`

### Labeled Outputs

Per-device labeled CSV files were generated:

```text
data/labeled_devices/device_1_labeled.csv
data/labeled_devices/device_2_labeled.csv
data/labeled_devices/device_3_labeled.csv
data/labeled_devices/device_4_labeled.csv
data/labeled_devices/device_5_labeled.csv
data/labeled_devices/device_6_labeled.csv
data/labeled_devices/device_7_labeled.csv
data/labeled_devices/device_8_labeled.csv
data/labeled_devices/device_9_labeled.csv
```

All device-level labeled CSVs were then combined into:

```text
data/labeled_devices/all_devices_labeled.csv
```

Combined labeled dataset size:

| Class | Rows |
|---|---:|
| benign | 555,932 |
| attack | 6,506,674 |
| total | 7,062,606 |

The dataset is highly imbalanced: attack traffic dominates the combined dataset.

## Train/Test Split

### Split Strategy

The combined dataset was split using stratified random sampling by `binary_target`.

Split ratio:

```text
train: 80%
test:  20%
```

Generated split files:

```text
data/splits/all_devices_train_stratified.csv
data/splits/all_devices_test_stratified.csv
```

### Split Distribution

| Split | Benign | Attack | Total |
|---|---:|---:|---:|
| Train | 444,746 | 5,205,339 | 5,650,085 |
| Test | 111,186 | 1,301,335 | 1,412,521 |
| Total | 555,932 | 6,506,674 | 7,062,606 |

The train and test class proportions are preserved by stratification.

### Important Methodology Note

This is a stratified random split. Because rows from the same `source_file` can appear in both train and test, the experiment measures performance under an in-distribution random split, not under a strict source-file or device-holdout generalization setting.

This is acceptable for a baseline comparison, but it should be stated clearly in any presentation or written report.

## Models and Experiment Design

All completed experiments use:

```text
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

The experiments intentionally use the same train/test split so that results are comparable.

### Experiment 1: RF Baseline With 115 Features

Purpose:

- Establish a strong baseline using all available feature columns.
- Measure binary detection performance without feature selection, scaling, or balancing.
- Produce feature importances for later top20 feature selection.

Configuration:

| Property | Value |
|---|---|
| Features | 115 |
| Balancing | none |
| Scaling | none |
| Feature selection | none |
| Train rows | 5,650,085 |
| Test rows | 1,412,521 |

Output files:

```text
outputs/reports/all_devices_stratified_random_forest_baseline.txt
outputs/reports/all_devices_stratified_random_forest_confusion_matrix.csv
outputs/reports/all_devices_stratified_random_forest_feature_importance.csv
```

### Experiment 2: RF With Top20 Features

Purpose:

- Test whether a much smaller feature set can preserve near-baseline performance.
- Reduce training time, model complexity, and feature dependency.

Top20 features were selected from the feature importance values of Experiment 1.

Configuration:

| Property | Value |
|---|---|
| Features | 20 |
| Balancing | none |
| Scaling | none |
| Feature selection | RF feature importance top20 |
| Train rows | 5,650,085 |
| Test rows | 1,412,521 |

Generated top20 split files:

```text
data/splits/all_devices_train_stratified_top20.csv
data/splits/all_devices_test_stratified_top20.csv
```

Output files:

```text
outputs/reports/all_devices_stratified_random_forest_top20.txt
outputs/reports/all_devices_stratified_random_forest_top20_confusion_matrix.csv
```

### Experiment 3: RF With Top20 Features + SMOTE

Purpose:

- Test whether balancing the highly imbalanced training set improves minority-class behavior.
- Apply SMOTE only to the training data, never to the test data.

Configuration:

| Property | Value |
|---|---|
| Features | 20 |
| Balancing | SMOTE on train only |
| Scaling | StandardScaler fit on train only |
| Feature selection | RF feature importance top20 |
| Train rows before SMOTE | 5,650,085 |
| Train rows after SMOTE | 10,410,678 |
| Test rows | 1,412,521 |

Training distribution before SMOTE:

| Class | Rows |
|---|---:|
| benign | 444,746 |
| attack | 5,205,339 |

Training distribution after SMOTE:

| Class | Rows |
|---|---:|
| benign | 5,205,339 |
| attack | 5,205,339 |

Output files:

```text
outputs/reports/all_devices_stratified_random_forest_top20_smote.txt
outputs/reports/all_devices_stratified_random_forest_top20_smote_confusion_matrix.csv
```

## Results

### Main Metrics

| Experiment | Features | Balancing | Scaling | Train Rows Used | Accuracy | Attack Precision | Attack Recall | Attack F1 | Benign Precision | Benign Recall | Benign F1 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RF baseline | 115 | none | none | 5,650,085 | 0.999999 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 0.999991 | 0.999996 |
| RF top20 | 20 | none | none | 5,650,085 | 0.999998 | 0.999998 | 0.999999 | 0.999999 | 0.999991 | 0.999982 | 0.999987 |
| RF top20 + SMOTE | 20 | SMOTE on train only | StandardScaler train only | 10,410,678 | 0.999999 | 0.999999 | 0.999999 | 0.999999 | 0.999991 | 0.999991 | 0.999991 |

### Runtime Comparison

| Experiment | Read Train Seconds | Read Test Seconds | Scale Seconds | SMOTE Seconds | Train Seconds | Predict Seconds |
|---|---:|---:|---:|---:|---:|---:|
| RF baseline | 30.49 | 8.28 | n/a | n/a | 418.56 | 1.05 |
| RF top20 | 4.81 | 1.16 | n/a | n/a | 82.31 | 0.42 |
| RF top20 + SMOTE | 5.27 | 1.28 | 0.81 | 110.73 | 260.80 | 0.35 |

### Error Counts

| Experiment | False Positives | False Negatives | Total Errors |
|---|---:|---:|---:|
| RF baseline | 1 | 0 | 1 |
| RF top20 | 2 | 1 | 3 |
| RF top20 + SMOTE | 1 | 1 | 2 |

Definitions:

- False positive: benign traffic predicted as attack.
- False negative: attack traffic predicted as benign.

In security detection, false negatives are usually more critical because they represent attacks that were missed.

### Confusion Matrices

#### RF Baseline, 115 Features

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 111,185 | 1 |
| attack | 0 | 1,301,335 |

#### RF Top20

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 111,184 | 2 |
| attack | 1 | 1,301,334 |

#### RF Top20 + SMOTE

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| benign | 111,185 | 1 |
| attack | 1 | 1,301,334 |

## Top20 Feature Set

The top20 feature set was selected from the 115-feature RF baseline feature importances.

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `HH_jit_L0.01_mean` | 0.141096 |
| 2 | `HH_L0.01_radius` | 0.073049 |
| 3 | `HpHp_L0.1_std` | 0.063822 |
| 4 | `HpHp_L0.01_radius` | 0.054188 |
| 5 | `HH_L0.01_covariance` | 0.052533 |
| 6 | `HpHp_L0.1_radius` | 0.052013 |
| 7 | `HpHp_L0.01_std` | 0.046935 |
| 8 | `MI_dir_L0.01_weight` | 0.042557 |
| 9 | `HH_L0.01_pcc` | 0.034466 |
| 10 | `HpHp_L0.01_weight` | 0.034144 |
| 11 | `HpHp_L0.1_weight` | 0.022801 |
| 12 | `HH_L0.1_pcc` | 0.022623 |
| 13 | `HH_jit_L0.1_variance` | 0.020559 |
| 14 | `HpHp_L1_radius` | 0.019928 |
| 15 | `HH_L0.1_covariance` | 0.016354 |
| 16 | `MI_dir_L0.1_weight` | 0.016181 |
| 17 | `HH_L0.1_radius` | 0.015778 |
| 18 | `HH_jit_L5_mean` | 0.015492 |
| 19 | `HH_L0.01_std` | 0.015332 |
| 20 | `H_L1_weight` | 0.013868 |

### Feature Pattern Observation

The most important features are concentrated around:

- `HH_*`: host-to-host traffic statistics
- `HpHp_*`: host-port to host-port traffic statistics
- short decay windows such as `L0.01` and `L0.1`
- jitter, radius, covariance, pcc, standard deviation, and weight statistics

This suggests that short-term traffic behavior and host/port interaction patterns are highly informative for binary botnet detection in this dataset.

## Interpretation

### RF Baseline Interpretation

The 115-feature Random Forest baseline achieved the strongest confusion matrix:

- 0 missed attacks
- 1 benign sample incorrectly flagged as attack
- attack recall of 1.000000

For a security detection problem, this is the most favorable result among the completed experiments because it avoids false negatives.

The downside is cost:

- highest feature count
- longest training time
- largest input representation

### RF Top20 Interpretation

The top20 RF model is the best compact model in the current experiment set.

Compared with the 115-feature baseline:

- feature count decreased from 115 to 20
- training time decreased from 418.56 seconds to 82.31 seconds
- training became about 5.1x faster
- total test errors increased from 1 to 3
- attack false negatives increased from 0 to 1

This is a strong compression result: 82.6% fewer features with almost identical metrics.

However, because this is a security use case, the one missed attack matters. If the priority is absolute attack coverage, the 115-feature baseline remains better.

### RF Top20 + SMOTE Interpretation

SMOTE balanced the training set by synthesizing benign samples until benign and attack counts were equal.

This changed the training distribution from:

```text
benign:   444,746
attack: 5,205,339
```

to:

```text
benign: 5,205,339
attack: 5,205,339
```

The result:

- false positives improved compared with top20 RF: 2 -> 1
- false negatives stayed at 1
- train time increased from 82.31 seconds to 260.80 seconds
- SMOTE itself added 110.73 seconds

SMOTE did not clearly improve attack detection in this binary setup. It may only be useful if reducing benign false positives is more important than training cost.

## Practical Model Choice

### If Accuracy and Attack Recall Are the Priority

Use:

```text
RF baseline with 115 features
```

Reason:

- no missed attacks in the test set
- only one false positive
- strongest raw detection result

### If Deployment Simplicity or Speed Is the Priority

Use:

```text
RF top20
```

Reason:

- much fewer features
- much faster training
- nearly identical performance
- simpler feature pipeline

### If Class Balancing Must Be Demonstrated

Use:

```text
RF top20 + SMOTE
```

Reason:

- shows a train-only balancing pipeline
- demonstrates correct handling of imbalance without touching the test set
- but should be presented as not clearly superior in this binary experiment

## Presentation-Ready Key Messages

The following points are suitable for slides or narration:

1. The full all-device labeled dataset contains 7,062,606 rows across nine IoT devices.
2. The binary target is highly imbalanced: 555,932 benign rows versus 6,506,674 attack rows.
3. A stratified 80/20 split was used to preserve the class distribution in train and test.
4. The 115-feature Random Forest baseline achieved near-perfect binary detection and missed zero attacks.
5. Feature importance from the baseline model was used to reduce the feature set from 115 to 20.
6. The top20 feature model trained about 5.1x faster while maintaining almost the same performance.
7. SMOTE balanced the training data but did not clearly improve attack recall.
8. The best raw model is the 115-feature RF baseline; the best lightweight model is RF top20.
9. Because this is a stratified random split, the result should be interpreted as in-distribution performance.
10. A stricter future evaluation would use source-file or device-holdout splitting.

## Suggested Slide Structure

### Slide 1: Experiment Objective

Main point:

- Detect benign vs attack IoT traffic using N-BaIoT network traffic features.

Suggested content:

- Dataset: N-BaIoT
- Scope: all nine devices
- Task: binary classification
- Models: Random Forest variants

### Slide 2: Data Labeling and Combination

Main point:

- Raw device/attack CSV files were labeled and combined into one all-device dataset.

Suggested content:

- benign files -> `binary_target = 0`
- Gafgyt/Mirai files -> `binary_target = 1`
- total rows: 7,062,606
- benign: 555,932
- attack: 6,506,674

### Slide 3: Train/Test Split

Main point:

- Stratified 80/20 split preserved class imbalance.

Suggested content:

- train rows: 5,650,085
- test rows: 1,412,521
- train benign/attack counts
- test benign/attack counts

### Slide 4: Experiment Setup

Main point:

- Three comparable Random Forest experiments were run on the same split.

Suggested content:

- RF baseline, 115 features
- RF top20
- RF top20 + SMOTE
- same random state and estimator count

### Slide 5: Results Table

Main point:

- All three models achieved extremely high performance.

Suggested content:

- accuracy
- attack precision
- attack recall
- attack F1
- training time

### Slide 6: Confusion Matrix Comparison

Main point:

- The 115-feature baseline had the cleanest test result.

Suggested content:

- baseline: 0 false negatives, 1 false positive
- top20: 1 false negative, 2 false positives
- top20 + SMOTE: 1 false negative, 1 false positive

### Slide 7: Feature Reduction Result

Main point:

- Top20 features preserve almost all performance while reducing training cost.

Suggested content:

- 115 -> 20 features
- 82.6% fewer features
- 418.56 sec -> 82.31 sec training
- about 5.1x faster

### Slide 8: SMOTE Result

Main point:

- SMOTE balanced the train data but did not clearly improve detection.

Suggested content:

- train rows after SMOTE: 10,410,678
- false positives improved by one compared with top20
- false negatives remained one
- training cost increased

### Slide 9: Caveat and Future Work

Main point:

- Stratified random split is useful for baseline comparison, but stricter validation is needed for generalization claims.

Suggested content:

- potential source-file leakage
- future source-file split
- future device-holdout split
- device-level error analysis

### Slide 10: Final Recommendation

Main point:

- Use the 115-feature baseline for best detection; use top20 RF for lightweight deployment.

Suggested content:

- best raw model: RF baseline
- best compact model: RF top20
- SMOTE not currently justified as default

## Limitations

### Stratified Random Split Limitation

The split is stratified by class, not by source file or device. This means the same traffic source file can contribute rows to both train and test. Since adjacent or related network traffic samples may be very similar, this can inflate performance.

This does not invalidate the comparison between the three experiments, because they all use the same split. However, it limits how strongly the results can be presented as cross-device or future-device generalization.

### Dataset Imbalance

The dataset contains many more attack rows than benign rows. This is why accuracy alone is not enough. Attack recall, benign recall, false positives, and false negatives should be reported together.

### SMOTE Cost

SMOTE increased the training set from 5,650,085 rows to 10,410,678 rows. This added computational cost without providing a clear improvement in attack detection.

### Binary-Only Scope

These experiments only evaluate binary classification:

```text
benign vs attack
```

They do not distinguish between Gafgyt and Mirai families or attack subtypes.

## Strict Cross-Device Validation: LODO Top20 RF

### Why LODO Was Added

The stratified random split results are useful for measuring in-distribution performance, but they are not the strictest validation strategy. Because the stratified split samples rows randomly, each device can contribute traffic to both train and test.

To test whether the model generalizes to a completely unseen device, a Leave-One-Device-Out (LODO) experiment was added.

LODO setup:

```text
Train on 8 devices
Test on the 1 held-out device
Repeat for all 9 devices
```

This is stricter and more deployment-like than the stratified split. It answers a different question:

```text
Can the model detect attacks on a device that was never included in training?
```

### LODO Model Setup

The LODO experiment was intentionally run only with the top20 feature set.

| Item | Value |
|---|---|
| Task | Binary classification: benign vs attack |
| Validation strategy | Leave-One-Device-Out |
| Folds | 9 |
| Model | RandomForestClassifier |
| Estimators | 100 |
| Random state | 42 |
| Features | Top20 RF-importance-selected features |
| Scaling | none |
| Balancing | none |

### LODO Overall Result

The LODO experiment evaluates every row once as part of a held-out device test fold.

| Metric | Value |
|---|---:|
| Total held-out test rows | 7,062,606 |
| Pooled accuracy | 0.999930 |
| Weighted attack recall | 0.999995 |
| Weighted benign recall | 0.999167 |
| Mean fold accuracy | 0.999932 |
| Mean fold attack recall | 0.999996 |
| Mean fold benign recall | 0.998911 |
| Mean fold macro F1 | 0.999685 |
| Total false positives | 463 |
| Total false negatives | 31 |

Security interpretation:

- Only 31 attack rows out of 6,506,674 attack rows were misclassified as benign.
- Attack recall stayed extremely high even when the test device was unseen during training.
- Most LODO errors were false positives, not missed attacks.

### LODO Per-Device Summary

| Held-Out Device | Device Name | Test Rows | Accuracy | Attack Recall | Benign Recall | FP | FN |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Danmini_Doorbell | 1,018,298 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 |
| 2 | Ecobee_Thermostat | 835,876 | 0.999999 | 1.000000 | 0.999924 | 1 | 0 |
| 3 | Ennio_Doorbell | 355,500 | 0.999949 | 1.000000 | 0.999540 | 18 | 0 |
| 4 | Philips_B120N10_Baby_Monitor | 1,098,677 | 0.999982 | 1.000000 | 0.999886 | 20 | 0 |
| 5 | Provision_PT_737E_Security_Camera | 828,260 | 0.999976 | 0.999974 | 1.000000 | 0 | 20 |
| 6 | Provision_PT_838_Security_Camera | 836,891 | 0.999994 | 1.000000 | 0.999949 | 5 | 0 |
| 7 | Samsung_SNH_1011_N_Webcam | 375,222 | 0.999984 | 1.000000 | 0.999885 | 6 | 0 |
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera | 863,056 | 0.999525 | 0.999999 | 0.991220 | 409 | 1 |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera | 850,826 | 0.999984 | 0.999988 | 0.999795 | 4 | 10 |

### LODO Error Analysis

Across all LODO folds:

```text
false positives: 463
false negatives: 31
total errors:    494
test rows:       7,062,606
```

The model produced many more false positives than false negatives:

```text
FP/FN ratio = 463 / 31 = 14.94
```

This means the top20 RF model tends to be conservative under unseen-device evaluation. It is more likely to flag uncertain benign traffic as attack than to miss attacks.

From a cybersecurity perspective, this is generally preferable to the opposite behavior, but false positives still matter operationally because they can create alert load.

### Hardest LODO Devices

Lowest attack recall:

| Device | Device Name | Attack Recall | False Negatives | Test Attack Rows |
|---:|---|---:|---:|---:|
| 5 | Provision_PT_737E_Security_Camera | 0.999974 | 20 | 766,106 |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera | 0.999988 | 10 | 831,298 |
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera | 0.999999 | 1 | 816,471 |

Lowest benign recall:

| Device | Device Name | Benign Recall | False Positives | Test Benign Rows |
|---:|---|---:|---:|---:|
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera | 0.991220 | 409 | 46,585 |
| 3 | Ennio_Doorbell | 0.999540 | 18 | 39,100 |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera | 0.999795 | 4 | 19,528 |

Key observations:

- Device 1 was classified perfectly under LODO.
- Device 8 is the main false-positive driver.
- Device 5 is the main false-negative driver.
- Even on the hardest attack-recall device, attack recall stayed at 0.999974.

### Stratified Top20 vs LODO Top20

The top20 RF model was evaluated under both stratified random split and LODO.

| Experiment | Test Strategy | Test Rows | Accuracy | Attack Recall | Benign Recall | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| Stratified top20 RF | Random stratified 80/20 | 1,412,521 | 0.999998 | 0.999999 | 0.999982 | 2 | 1 |
| LODO top20 RF | One held-out device per fold | 7,062,606 pooled | 0.999930 | 0.999995 | 0.999167 | 463 | 31 |

Interpretation:

- LODO is harder and more realistic for unseen-device deployment.
- Performance drops slightly compared with stratified random split, as expected.
- The drop is mostly visible in benign recall because of false positives on Device 8.
- Attack recall remains extremely high under both evaluation strategies.

### LODO Takeaway

The LODO result strengthens the top20 RF conclusion.

The top20 feature model is not only strong under stratified random evaluation; it also generalizes very well to unseen devices. This makes it a credible lightweight candidate for binary IoT botnet detection.

However, the device-level error distribution matters:

- operational tuning should focus on Device 8 false positives
- security-risk inspection should focus on Device 5 and Device 9 false negatives
- future reporting should separate stratified results from LODO results because they measure different forms of performance

## Reproducibility

### Scripts Used

Labeling:

```text
scripts/export_labeled_devices.py
```

Stratified split:

```text
scripts/split_all_devices_stratified.py
```

115-feature RF baseline:

```text
scripts/train_all_devices_stratified_baseline_rf.py
```

Top20 split creation:

```text
scripts/create_all_devices_stratified_top20_splits.py
```

Top20 RF:

```text
scripts/train_all_devices_stratified_top20_rf.py
```

Top20 + SMOTE RF:

```text
scripts/train_all_devices_stratified_top20_smote_rf.py
```

LODO top20 RF:

```text
scripts/train_lodo_top20_rf.py
```

### Main Output Files

```text
outputs/reports/all_devices_stratified_random_forest_baseline.txt
outputs/reports/all_devices_stratified_random_forest_confusion_matrix.csv
outputs/reports/all_devices_stratified_random_forest_feature_importance.csv
outputs/reports/all_devices_stratified_random_forest_top20.txt
outputs/reports/all_devices_stratified_random_forest_top20_confusion_matrix.csv
outputs/reports/all_devices_stratified_random_forest_top20_smote.txt
outputs/reports/all_devices_stratified_random_forest_top20_smote_confusion_matrix.csv
outputs/reports/lodo_top20_random_forest_report.md
outputs/reports/lodo_top20_random_forest_results.csv
outputs/reports/lodo_top20_confusion_matrices/
```

## Final Conclusion

The all-device binary classification pipeline is complete for the current scope, including both in-distribution stratified testing and stricter cross-device LODO validation.

The 115-feature Random Forest baseline is the best-performing model in terms of missed attacks and overall confusion matrix quality. It should be treated as the strongest current result.

The top20 Random Forest model is the most practical lightweight alternative. It dramatically reduces feature count and training time while preserving almost the same performance.

The top20 + SMOTE model demonstrates a correct train-only balancing workflow, but it does not clearly outperform the simpler top20 RF model in a way that justifies the added cost.

The LODO top20 RF experiment shows that the compact model also generalizes strongly to unseen devices. Across all nine held-out device folds, it reached 0.999930 pooled accuracy and 0.999995 weighted attack recall, with only 31 missed attacks across 6,506,674 attack rows.

For presentation purposes, the main story is:

```text
Full-feature RF gives the cleanest detection.
Top20 RF gives nearly the same detection much faster.
SMOTE is methodologically correct but not clearly beneficial here.
LODO confirms that top20 RF generalizes strongly to unseen devices.
```
