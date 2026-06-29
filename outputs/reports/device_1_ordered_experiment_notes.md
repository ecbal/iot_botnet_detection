# Device 1 Ordered Split Deney Notları

## Split Stratejisi

Bu deneylerde cihaz 1 için random stratified split yerine source bazlı sıralı split kullanıldı.

Her `source_file` kendi içinde ayrıldı:

```text
ilk %80 -> train
son %20 -> test
```

Bu sayede train ve test içinde her trafik tipi bulunur, fakat aynı source dosya içindeki sıra tamamen rastgele karıştırılmaz.

Dosyalar:

```text
data/splits/device_1_train_ordered_by_source.csv
data/splits/device_1_test_ordered_by_source.csv
```

Dağılım:

```text
Train rows: 814,635
attack:     774,997
benign:      39,638

Test rows: 203,663
attack:     193,753
benign:       9,910
```

## RF Baseline, 115 Feature

Rapor:

```text
outputs/reports/device_1_ordered_random_forest_baseline.txt
```

Sonuç:

```text
Accuracy:         1.000000
Attack precision: 1.000000
Attack recall:    1.000000
Attack F1:        1.000000
Benign precision: 1.000000
Benign recall:    1.000000
Benign F1:        1.000000
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign              9910                 0
actual_attack                 0            193753
```

Eğitim süresi:

```text
15.86 sec
```

## Ordered Top20 Feature Seti

Top20 feature listesi ordered RF baseline modelinin feature importance skorlarından üretildi.

Dosyalar:

```text
data/splits/device_1_train_ordered_top20.csv
data/splits/device_1_test_ordered_top20.csv
```

Seçilen feature'lar:

```text
HH_jit_L0.01_mean
HpHp_L0.1_weight
H_L0.01_weight
HH_L0.01_radius
HH_jit_L3_mean
HpHp_L0.01_weight
MI_dir_L0.1_weight
MI_dir_L0.1_mean
HH_L0.01_pcc
HH_L0.01_covariance
MI_dir_L1_weight
HH_L0.01_magnitude
HH_jit_L5_mean
HH_jit_L1_mean
MI_dir_L3_weight
H_L0.1_weight
HH_L0.01_weight
MI_dir_L0.01_variance
HH_L0.01_std
MI_dir_L1_mean
```

## RF Top20

Rapor:

```text
outputs/reports/device_1_ordered_random_forest_top20.txt
```

Sonuç:

```text
Accuracy:         1.000000
Attack precision: 1.000000
Attack recall:    1.000000
Attack F1:        1.000000
Benign precision: 1.000000
Benign recall:    1.000000
Benign F1:        1.000000
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign              9910                 0
actual_attack                 0            193753
```

Eğitim süresi:

```text
8.74 sec
```

Top20 feature ile 115 feature performansı korundu ve eğitim süresi düştü.

## KNN Top20

Rapor:

```text
outputs/reports/device_1_ordered_knn_top20.txt
```

KNN için `StandardScaler` sadece train set üzerinde fit edildi.

Sonuç:

```text
Accuracy:         0.999995
Attack precision: 1.000000
Attack recall:    0.999995
Attack F1:        0.999997
Benign precision: 0.999899
Benign recall:    1.000000
Benign F1:        0.999950
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign              9910                 0
actual_attack                 1            193752
```

Süre:

```text
Fit:     0.03 sec
Predict: 117.40 sec
```

KNN sadece 1 hata yaptı, fakat tahmin süresi RF'ye göre çok yüksek.

## RF Top20 + SMOTE

Rapor:

```text
outputs/reports/device_1_ordered_random_forest_top20_smote.txt
```

SMOTE sadece train set üzerinde uygulandı. Test setine sentetik örnek eklenmedi.

Train dağılımı SMOTE öncesi:

```text
benign:  39,638
attack: 774,997
```

Train dağılımı SMOTE sonrası:

```text
benign: 774,997
attack: 774,997
```

Sonuç:

```text
Accuracy:         1.000000
Attack precision: 1.000000
Attack recall:    1.000000
Attack F1:        1.000000
Benign precision: 1.000000
Benign recall:    1.000000
Benign F1:        1.000000
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign              9910                 0
actual_attack                 0            193753
```

Süre:

```text
SMOTE:   2.44 sec
Train:  44.75 sec
Predict: 0.08 sec
```

SMOTE performansı bozmadı, fakat RF top20 zaten hatasız olduğu için ek fayda göstermedi. Eğitim süresi belirgin şekilde arttı.

## Karşılaştırma

```text
RF 115 feature:
Accuracy: 1.000000
Hata: 0
Train: 15.86 sec
Predict: 0.09 sec

RF top20:
Accuracy: 1.000000
Hata: 0
Train: 8.74 sec
Predict: 0.06 sec

KNN top20:
Accuracy: 0.999995
Hata: 1
Fit: 0.03 sec
Predict: 117.40 sec

RF top20 + SMOTE:
Accuracy: 1.000000
Hata: 0
SMOTE: 2.44 sec
Train: 44.75 sec
Predict: 0.08 sec
```

## Yorum

Ordered split, random stratified split'e göre daha kontrollü bir senaryo olmasına rağmen cihaz 1 binary classification problemi hala çok kolay görünüyor.

Bu deneylerde en pratik model:

```text
Random Forest + ordered top20 feature
```

Çünkü:

```text
0 hata
20 feature
SMOTE yok
GAN yok
daha kısa eğitim süresi
hızlı tahmin
```

GAN deneyi yapılabilir, ancak cihaz 1 için mevcut sonuçlar tavan seviyede olduğu için GAN'ın performans kazancı göstermesi beklenmez. GAN daha çok yöntem karşılaştırması veya başka cihazlarda daha zor splitlerde anlamlı olabilir.
