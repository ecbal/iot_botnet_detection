# Device 1 Train -> Device 2 Full Test Notları

## Amaç

Bu deney ezberleme riskini daha iyi kontrol etmek için yapıldı.

Önceki deneylerde train ve test aynı cihazdan geliyordu. Bu deneyde ise:

```text
train: device 1 ordered train set
test:  device 2'nin tamamı
```

Device 2 bölünmedi. Tüm device 2 labeled datası test seti olarak kullanıldı.

## Device 2 Labeling

Device 2 için tüm CSV dosyaları birleştirildi:

```text
data/labeled_devices/device_2_labeled.csv
```

Label dağılımı:

```text
attack    822,763
benign     13,113
```

Kaynak dosyalar:

```text
2.benign.csv             13,113
2.gafgyt.combo.csv       53,012
2.gafgyt.junk.csv        30,312
2.gafgyt.scan.csv        27,494
2.gafgyt.tcp.csv         95,021
2.gafgyt.udp.csv        104,791
2.mirai.ack.csv         113,285
2.mirai.scan.csv         43,192
2.mirai.syn.csv         116,807
2.mirai.udp.csv         151,481
2.mirai.udpplain.csv     87,368
```

## Denenen Modeller

Üç RF varyantı denendi:

```text
1. RF 115 feature
2. RF top20 feature
3. RF top20 feature + SMOTE
```

Top20 feature listesi device 1 ordered RF baseline feature importance skorlarından alındı.

## Sonuçlar

### RF 115 Feature

```text
Accuracy:         0.999946
Attack precision: 0.999945
Attack recall:    1.000000
Attack F1:        0.999973
Benign precision: 1.000000
Benign recall:    0.996568
Benign F1:        0.998281
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign             13068                45
actual_attack                 0            822763
```

### RF Top20 Feature

```text
Accuracy:         0.999970
Attack precision: 0.999970
Attack recall:    1.000000
Attack F1:        0.999985
Benign precision: 1.000000
Benign recall:    0.998093
Benign F1:        0.999046
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign             13088                25
actual_attack                 0            822763
```

### RF Top20 + SMOTE

```text
Accuracy:         0.999614
Attack precision: 0.999608
Attack recall:    1.000000
Attack F1:        0.999804
Benign precision: 1.000000
Benign recall:    0.975368
Benign F1:        0.987530
```

Confusion matrix:

```text
               predicted_benign  predicted_attack
actual_benign             12790               323
actual_attack                 0            822763
```

## Karşılaştırma

```text
RF 115:
Hata: 45
Attack kaçırma: 0

RF top20:
Hata: 25
Attack kaçırma: 0

RF top20 + SMOTE:
Hata: 323
Attack kaçırma: 0
```

Bu cross-device testte en iyi sonuç:

```text
Random Forest + top20 feature
```

## Yorum

Device 1'de eğitilip device 2'nin tamamında test edildiğinde sonuçlar hala çok yüksek çıktı. Bu, modelin sadece aynı cihaz içindeki random/ordered split'i ezberlemediğine dair daha güçlü bir işarettir.

Fakat bu hala tamamen nihai genelleme kanıtı değildir. Çünkü device 1 ve device 2 aynı dataset yapısından, aynı feature extractor'dan ve aynı saldırı ailelerinden geliyor.

Daha güçlü testler:

```text
1. Train device 1, test device 3/4/5...
2. Birden fazla cihazda eğit, tamamen ayrı cihazda test et.
3. Mirai holdout veya Gafgyt holdout gibi attack family bazlı test yap.
4. Attack type holdout: train'de olmayan saldırı tipini testte yakalamaya çalış.
```

Bu aşamada device 2 testi, ezberleme şüphesini azaltıyor ama tamamen ortadan kaldırmıyor.
