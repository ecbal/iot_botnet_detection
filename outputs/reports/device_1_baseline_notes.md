# Device 1 Binary Baseline Notları

## Deney Özeti

Bu deneyde sadece cihaz 1 verisi kullanıldı. Cihaz 1'e ait tüm CSV dosyaları önce tek bir labeled CSV dosyasında birleştirildi:

```text
data/labeled_devices/device_1_labeled.csv
```

Ardından veri stratified random split ile train/test olarak ayrıldı:

```text
data/splits/device_1_train.csv
data/splits/device_1_test.csv
```

Split oranı:

```text
train: %80
test:  %20
```

## Label Bilgisi

Binary label mapping:

```text
benign -> binary_target = 0
attack -> binary_target = 1
```

Cihaz 1 toplam dağılımı:

```text
attack    968,750
benign     49,548
```

Train dağılımı:

```text
attack    775,000
benign     39,638
```

Test dağılımı:

```text
attack    193,750
benign      9,910
```

## Baseline Model

Kullanılan model:

```text
RandomForestClassifier
```

Model ayarları:

```text
n_estimators = 100
random_state = 42
class balancing = none
feature selection = none
scaling = none
```

Bu deneyde özellikle hiçbir iyileştirme yapılmadı. Amaç ham durumda modelin performansını görmekti.

## Sonuçlar

Test sonuçları:

```text
Accuracy:         0.999980
Attack precision: 0.999995
Attack recall:    0.999985
Attack F1:        0.999990
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign              9909                 1
actual_attack                 3            193747
```

Test setinde toplam 4 hata yapıldı:

```text
benign -> attack: 1
attack -> benign: 3
```

## İlk 20 Feature Importance

Random Forest'a göre en önemli ilk 20 feature:

```text
HH_jit_L0.01_mean       0.128620
HpHp_L0.1_weight        0.057630
MI_dir_L0.1_weight      0.050624
HH_L0.01_radius         0.046589
HpHp_L0.01_weight       0.039560
HH_jit_L3_mean          0.033017
H_L0.01_weight          0.031008
H_L0.1_weight           0.030083
MI_dir_L0.1_mean        0.029284
HH_L0.01_pcc            0.028314
HH_L0.01_covariance     0.027930
HH_L0.01_magnitude      0.027025
HH_jit_L1_mean          0.026970
HH_L1_weight            0.019350
MI_dir_L0.1_variance    0.018464
MI_dir_L0.01_weight     0.017625
H_L1_weight             0.017595
HH_jit_L5_mean          0.017581
HH_L0.01_std            0.016510
H_L0.01_variance        0.016300
```

## Yorum

Baseline performansı çok yüksek çıktı. Bu ilk aşama için iyi bir referans noktasıdır, fakat sonucun bu kadar yüksek olması dikkatli yorumlanmalıdır.

Mevcut split aynı cihazın aynı trafik kaynaklarından rastgele örnekler alıyor. Bu nedenle train ve test dağılımları birbirine çok benzer olabilir. Gerçek hayatta daha zor senaryolar için ilerleyen aşamalarda farklı split stratejileri denenebilir:

```text
source_file bazlı split
zaman/sıra bazlı split
cihazlar arası genelleme testi
```

Şu anki sonuç "ham durumda cihaz 1 üzerinde Random Forest nasıl performans veriyor?" sorusuna cevap verir. Bundan sonraki SMOTE, feature selection veya farklı model denemeleri bu baseline ile karşılaştırılmalıdır.

## RF Top 20 Feature Deneyi

Baseline sonrasında Random Forest feature importance skorlarına göre ilk 20 feature seçildi. Train ve test dosyaları bu 20 feature'a indirildi:

```text
data/splits/device_1_train_top20.csv
data/splits/device_1_test_top20.csv
```

Bu dosyalarda şu kolonlar bulunur:

```text
20 feature
binary_label
binary_target
source_file
```

Kullanılan model:

```text
RandomForestClassifier
n_estimators = 100
random_state = 42
class balancing = none
feature selection = RF feature importance top20
scaling = none
```

Test sonuçları:

```text
Accuracy:         0.999985
Attack precision: 1.000000
Attack recall:    0.999985
Attack F1:        0.999992
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign              9910                 0
actual_attack                 3            193747
```

Karşılaştırma:

```text
RF 115 feature:
Accuracy: 0.999980
Hata: 4

RF top20 feature:
Accuracy: 0.999985
Hata: 3
```

Bu sonuç cihaz 1 için ilk 20 feature'ın performansı düşürmediğini gösteriyor. Hatta bu split üzerinde 115 feature baseline'dan çok az daha iyi sonuç verdi. Eğitim süresi de azaldı:

```text
RF 115 feature train time: 21.72 seconds
RF top20 train time:       9.83 seconds
```

Bu nedenle cihaz 1 binary classification için top20 feature seti kullanılabilir görünüyor. Sonraki adım aynı top20 feature setiyle KNN modelini denemek olabilir.

## KNN Top 20 Feature Deneyi

Random Forest ile seçilen aynı top20 feature seti KNN modeliyle de denendi.

KNN mesafe tabanlı bir model olduğu için feature'lar önce ölçeklendirildi:

```text
scaler = StandardScaler
fit: sadece train set
transform: train ve test set
```

Kullanılan model:

```text
KNeighborsClassifier
n_neighbors = 5
weights = distance
algorithm = auto
class balancing = none
feature selection = RF feature importance top20
```

Test sonuçları:

```text
Accuracy:         0.999936
Attack precision: 0.999985
Attack recall:    0.999948
Attack F1:        0.999966
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign              9907                 3
actual_attack                10            193740
```

Test setinde toplam 13 hata yapıldı:

```text
benign -> attack: 3
attack -> benign: 10
```

Karşılaştırma:

```text
RF 115 feature:
Accuracy: 0.999980
Hata: 4
Train time: 21.72 sec
Predict time: 0.09 sec

RF top20 feature:
Accuracy: 0.999985
Hata: 3
Train time: 9.83 sec
Predict time: 0.06 sec

KNN top20 feature:
Accuracy: 0.999936
Hata: 13
Fit time: 0.03 sec
Predict time: 67.49 sec
```

KNN sonucu oldukça iyi olmasına rağmen RF top20 sonucundan daha zayıf kaldı. Ayrıca KNN'de fit süresi çok kısa olsa da predict süresi çok daha uzundur, çünkü her test örneği için train setindeki komşular aranır.

Bu aşamada cihaz 1 binary classification için en iyi aday:

```text
Random Forest + top20 feature
```

## RF Top 20 + SMOTE Deneyi

Bu deneyde class balancing için SMOTE kullanıldı.

Önemli veri sızıntısı kuralı:

```text
SMOTE sadece train set üzerinde uygulandı.
Test set üzerinde SMOTE uygulanmadı.
```

KNN gibi SMOTE da komşuluk/mesafe tabanlı çalıştığı için feature'lar önce ölçeklendirildi:

```text
scaler = StandardScaler
fit: sadece train set
transform: train ve test set
```

Train dağılımı SMOTE öncesi:

```text
benign:  39,638
attack: 775,000
```

Train dağılımı SMOTE sonrası:

```text
benign: 775,000
attack: 775,000
```

Test dağılımı değiştirilmedi:

```text
benign:   9,910
attack: 193,750
```

Kullanılan model:

```text
RandomForestClassifier
n_estimators = 100
random_state = 42
class balancing = SMOTE on train only
feature selection = RF feature importance top20
scaling = StandardScaler fit on train only
```

Test sonuçları:

```text
Accuracy:         0.999971
Attack precision: 0.999995
Attack recall:    0.999974
Attack F1:        0.999985
Benign precision: 0.999496
Benign recall:    0.999899
Benign F1:        0.999697
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign              9909                 1
actual_attack                 5            193745
```

Karşılaştırma:

```text
RF 115 feature:
Accuracy: 0.999980
Hata: 4

RF top20 feature:
Accuracy: 0.999985
Hata: 3

KNN top20 feature:
Accuracy: 0.999936
Hata: 13

RF top20 + SMOTE:
Accuracy: 0.999971
Hata: 6
```

Bu split üzerinde SMOTE, RF top20 sonucunu iyileştirmedi. Bunun nedeni baseline performansının zaten neredeyse tavan seviyede olması olabilir. SMOTE özellikle minority class recall düşük olduğunda daha anlamlı katkı sağlar; burada benign recall zaten çok yüksekti.

Bu aşamadaki en iyi sonuç hala:

```text
Random Forest + top20 feature, SMOTE olmadan
```
