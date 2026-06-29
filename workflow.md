# NTA ile IoT Botnet Tespiti Workflow

## Genel Amaç

Bu projede N-BaIoT datasetini kullanarak IoT cihazlardaki botnet trafiğini Network Traffic Analysis (NTA) yaklaşımıyla tespit eden machine learning modelleri geliştirilecek. Temel hedef, ağ trafiğinden çıkarılmış 115 sayısal feature üzerinden benign ve saldırı trafiğini ayırt etmek; ek olarak saldırı tiplerini daha detaylı sınıflandırabilecek multiclass modelleri de denemektir.

Dataset 9 farklı IoT cihazdan toplanmış gerçek trafik kayıtlarını içerir. Trafikler benign, Gafgyt/BASHLITE ve Mirai botnet saldırılarına aittir. Bu yüzden proje hem binary classification hem multiclass classification hem de anomaly detection/unsupervised detection açısından çalışmaya uygundur.

## Dataset

Kullanılacak veri dizini:

```text
archive-2/
```

Önemli dosyalar:

- `archive-2/*.csv`: Cihaz ve trafik tipine göre ayrılmış veri dosyaları.
- `archive-2/data_summary.csv`: Her CSV dosyasının satır ve feature sayısı.
- `archive-2/device_info.csv`: Cihaz ID ve cihaz adı eşleşmeleri.
- `archive-2/features.csv`: 115 feature için açıklamalar.
- `archive-2/README.md`: Dataset açıklaması ve referans bilgileri.

Dosya adları label üretmek için kullanılacak:

- `*.benign.csv`: normal trafik.
- `*.gafgyt.combo.csv`, `*.gafgyt.junk.csv`, `*.gafgyt.scan.csv`, `*.gafgyt.tcp.csv`, `*.gafgyt.udp.csv`: Gafgyt/BASHLITE saldırıları.
- `*.mirai.ack.csv`, `*.mirai.scan.csv`, `*.mirai.syn.csv`, `*.mirai.udp.csv`, `*.mirai.udpplain.csv`: Mirai saldırıları.

## Problem Tanımları

Projede üç deney hattı kurulacak:

1. Binary classification:
   - `benign` ve `attack` ayrımı yapılacak.
   - Bütün Gafgyt ve Mirai dosyaları `attack` label'ına indirgenecek.

2. Multiclass classification:
   - `benign` ayrı sınıf olacak.
   - Saldırılar alt tiplerine göre ayrılacak.
   - Örnek sınıflar: `gafgyt_combo`, `gafgyt_junk`, `gafgyt_scan`, `gafgyt_tcp`, `gafgyt_udp`, `mirai_ack`, `mirai_scan`, `mirai_syn`, `mirai_udp`, `mirai_udpplain`.

3. Unsupervised/anomaly detection:
   - Normal davranışı öğrenmek için benign trafik kullanılacak.
   - Test aşamasında benign ve attack trafiği birlikte değerlendirilecek.
   - Isolation Forest, One-Class SVM veya Autoencoder gibi yöntemler karşılaştırılabilir.

## Temel Deney Kuralı

Veri sızıntısını engellemek için işlem sırası şu şekilde olacak:

1. CSV dosyaları okunur.
2. Label ve device bilgisi üretilir.
3. Train/test split yapılır.
4. Balancing sadece train set üzerinde yapılır.
5. Feature selection sadece train set üzerinde fit edilir.
6. Test set üzerinde balancing yapılmaz.
7. Test set sadece train'de öğrenilen scaler/selector ile transform edilir.

Planlanan split:

```text
train: %80
test:  %20
```

Bu kural özellikle SMOTE, GAN tabanlı veri üretimi ve feature selection için geçerlidir. Test datasına hiçbir şekilde sentetik örnek eklenmeyecek ve test datası feature seçimi için kullanılmayacaktır.

## Veri Hazırlama Adımları

1. Dosya keşfi:
   - `archive-2/` altındaki tüm trafik CSV dosyaları listelenir.
   - `README.md`, `data_summary.csv`, `device_info.csv`, `features.csv` metadata olarak ayrılır.

2. Label parsing:
   - Dosya adındaki ilk sayı `device_id` olarak alınır.
   - Dosya adındaki kalan bölüm trafik/saldırı tipi olarak alınır.
   - Binary label için `benign` dışındaki her şey `attack` kabul edilir.
   - Multiclass label için saldırı ailesi ve saldırı tipi korunur.

3. Veri birleştirme:
   - Her CSV okunur.
   - Her satıra `device_id`, `device_name`, `binary_label`, `multiclass_label` eklenir.
   - Büyük veri boyutu nedeniyle gerekirse chunk bazlı okuma veya cihaz bazlı deney yapılır.

4. Temizlik:
   - Eksik değer kontrolü yapılır.
   - Infinite değer kontrolü yapılır.
   - Duplicate satırlar raporlanır.
   - Feature kolonlarının tamamının sayısal olduğu doğrulanır.

5. Ölçekleme:
   - StandardScaler veya MinMaxScaler train set üzerinde fit edilir.
   - Test set aynı scaler ile transform edilir.

## Class Balancing

Dataset saldırı tipleri ve cihazlar arasında dengesiz olabilir. Bu yüzden balancing yöntemleri train set üzerinde denenecek:

1. Baseline:
   - Hiç balancing yapılmadan model eğitilir.
   - Diğer yöntemler için referans skor oluşturur.

2. SMOTE:
   - Sadece train set üzerinde uygulanır.
   - Binary ve multiclass senaryolar için ayrı değerlendirilir.

3. GAN tabanlı balancing:
   - Sentetik örnek üretimi için GAN yaklaşımı denenir.
   - GAN sonucu test set üzerinde değil, yalnızca train set üzerinde kullanılacaktır.
   - SMOTE ile karşılaştırmalı raporlanır.

Not: GAN yöntemi daha iyi sonuç verebilir, fakat bu sonuç deneyle doğrulanacak. Test tarafı hiçbir balancing işlemine dahil edilmeyecek.

## Feature Selection

Amaç, 115 feature içinden en anlamlı ilk 20 feature'ı seçerek daha hafif ve yorumlanabilir modeller üretmektir.

Denenecek yöntemler:

1. Random Forest feature importance:
   - RF modeli train set üzerinde fit edilir.
   - En yüksek importance değerine sahip ilk 20 feature seçilir.

2. KNN tabanlı değerlendirme:
   - Seçilen feature setleri KNN modeliyle test edilir.
   - KNN doğrudan feature importance üretmediği için wrapper/filter yaklaşımıyla karşılaştırmalı kullanılabilir.

Feature selection akışı:

1. Selector sadece train set üzerinde fit edilir.
2. İlk 20 feature belirlenir.
3. Train ve test set aynı feature listesiyle daraltılır.
4. Seçilen feature isimleri rapora yazılır.

## Modelleme Planı

Supervised modeller:

- Random Forest
- KNN
- Logistic Regression
- SVM
- XGBoost veya LightGBM
- MLP

Unsupervised/anomaly detection modeller:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- Autoencoder

Deney kombinasyonları:

- Binary baseline
- Binary + SMOTE
- Binary + GAN balancing
- Binary + RF top 20 feature
- Multiclass baseline
- Multiclass + SMOTE
- Multiclass + GAN balancing
- Multiclass + RF top 20 feature
- Device bazlı anomaly detection

## Değerlendirme Metrikleri

Binary classification:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Multiclass classification:

- Macro F1
- Weighted F1
- Per-class precision/recall/F1
- Confusion matrix

Unsupervised/anomaly detection:

- Precision
- Recall
- F1-score
- ROC-AUC veya PR-AUC
- False positive rate

IoT botnet tespitinde recall ve false negative oranı özellikle önemli olacak. Attack trafiğini benign olarak kaçırmak güvenlik açısından daha kritik bir hata kabul edilecek.

## Önerilen Proje Yapısı

```text
.
├── archive-2/
├── workflow.md
├── main.py
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── balancing.py
│   ├── feature_selection.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_binary_classification.ipynb
│   ├── 03_multiclass_classification.ipynb
│   └── 04_anomaly_detection.ipynb
├── outputs/
│   ├── models/
│   ├── reports/
│   └── figures/
└── requirements.txt
```

## İlk Uygulama Sırası

1. Dataset dosyalarını otomatik okuyacak loader yaz.
2. Dosya adından `device_id`, `attack_family`, `attack_type`, `binary_label`, `multiclass_label` üret.
3. EDA raporu çıkar:
   - Toplam satır sayısı.
   - Cihaz başına kayıt sayısı.
   - Binary class dağılımı.
   - Multiclass dağılımı.
   - Feature istatistikleri.
4. Baseline binary model eğit:
   - Önce balancing ve feature selection olmadan.
5. Train/test split sonrası SMOTE ekle.
6. RF feature importance ile ilk 20 feature'ı seç.
7. Aynı deneyleri multiclass sınıflandırmaya taşı.
8. GAN balancing yöntemini ayrı deney olarak ekle.
9. Unsupervised/anomaly detection hattını kur.
10. Sonuçları tek raporda karşılaştır.

## Raporlanacak Çıktılar

Her deney için aşağıdakiler kaydedilecek:

- Kullanılan problem tipi: binary, multiclass veya unsupervised.
- Kullanılan balancing yöntemi: none, SMOTE veya GAN.
- Kullanılan feature selection yöntemi.
- Seçilen feature listesi.
- Model hiperparametreleri.
- Confusion matrix.
- Classification report.
- F1, recall, precision ve accuracy skorları.
- Eğitim süresi ve tahmin süresi.

## Dikkat Edilecek Noktalar

- Test datasına SMOTE veya GAN uygulanmayacak.
- Feature selection test datasında fit edilmeyecek.
- Scaler test datasında fit edilmeyecek.
- Cihaz bazlı sonuçlar ayrıca incelenecek; model bazı cihazlarda iyi, bazılarında kötü performans verebilir.
- Binary sonuçlar yüksek çıksa bile multiclass sonuçlar ayrıca değerlendirilecek.
- Sınıf dengesizliği nedeniyle sadece accuracy yeterli kabul edilmeyecek.
- Büyük CSV dosyaları belleği zorlayabilir; gerekirse cihaz bazlı veya chunk bazlı çalışma yapılacak.

## Beklenen Sonuç

Proje sonunda N-BaIoT dataseti üzerinde NTA feature'larıyla çalışan, IoT botnet trafiğini tespit edebilen ve farklı deney senaryolarında karşılaştırılmış bir machine learning pipeline'ı elde edilecek. Nihai raporda baseline, balancing yöntemleri, feature selection ve model türleri karşılaştırılarak en uygun yaklaşım belirlenecek.
