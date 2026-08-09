# Müşteri Ayrılma (Churn) Tahmini

Türkiye Yapay Zeka Akademisi - Makine Öğrenmesi Ara Ödevi

## Projenin Amacı

Müşteri bilgilerinden (yaş, gelir, abonelik süresi, destek talebi sayısı, şehir, üyelik tipi)
müşterinin hizmeti bırakıp bırakmayacağını (churn) tahmin eden basit bir sınıflandırma akışı
kurmak. Akışın tamamı tek Python dosyasında: veri okuma, temel veri inceleme, eksik değer
doldurma, öznitelik üretme, one-hot encoding, train/validation/test bölme, ölçekleme,
model karşılaştırma ve test değerlendirmesi.

## Veri Seti

Derste churn senaryosuna uygun hazır bir veri seti paylaşılmadığı için şartnamedeki
"en az 100 satırlık veri üretme" seçeneğini kullandım. `veri_seti_olustur.py` sabit seed ile
300 satırlık sentetik müşteri verisi üretiyor; üretilen `musteri_churn_verisi.csv` repoda hazır
olarak var. Gerçekçi olması için churn olasılığını kurallara bağladım (destek talebi arttıkça
ve abonelik süresi kısaldıkça risk artıyor, Bronz üyeler daha riskli) ve gelir ile üyelik
tipine bilerek az sayıda eksik değer ekledim.

## Nasıl Çalıştırılır

```bash
pip install -r requirements.txt
python3 musteri_churn_tahmini.py
```

CSV dosyası silinirse `python3 veri_seti_olustur.py` ile aynı veri tekrar üretilebilir
(seed sabit olduğu için sonuçlar değişmez).

## Dosyalar

- `musteri_churn_tahmini.py` : Ana akış (veri inceleme -> ön işleme -> modeller -> değerlendirme)
- `veri_seti_olustur.py` : Sentetik veri setini üreten script
- `musteri_churn_verisi.csv` : Kullanılan veri seti (300 satır)
- `requirements.txt` : Gerekli kütüphaneler

## Sonuçlar ve Kısa Yorum

Modeller validation seti üzerindeki F1 skoruna göre karşılaştırıldı:

- Logistic Regression : Validation F1 = 0.333
- KNN (k=5) : Validation F1 = 0.400
- Decision Tree (bonus) : Validation F1 = 0.273

En iyi model KNN (k=5) seçildi ve test setinde değerlendirildi:
Accuracy = 0.667, Precision = 0.438, Recall = 0.389, F1 = 0.412.

Churn eden müşteriler azınlıkta olduğu için (%30) accuracy tek başına yanıltıcı olabilirdi,
bu yüzden model seçiminde F1 skorunu baz aldım. KNN'in önde olmasının nedeni bence ölçekleme
sonrası benzer profildeki müşterilerin uzayda birbirine yakın düşmesi; yine de veri seti küçük
olduğu için modeller arasındaki farklar çok büyük değil. Skorların genel olarak orta seviyede
kalması normal, çünkü sentetik veride churn'ü belirleyen sinyal sınırlı ve bir miktar
rastgelelik var.
