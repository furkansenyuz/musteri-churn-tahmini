"""
Müşteri Ayrılma (Churn) Tahmini - Makine Öğrenmesi Ara Ödevi

Amaç:
Müşteri bilgilerini kullanarak müşterinin hizmeti bırakıp bırakmayacağını (churn)
tahmin eden basit bir sınıflandırma akışı kurmak. Akış sırasıyla: veri okuma,
temel veri inceleme, eksik değer doldurma, öznitelik üretme, one-hot encoding,
train/validation/test bölme, ölçekleme, model karşılaştırma ve test değerlendirmesi.

Kullanılan Kütüphaneler:
- pandas, numpy : Veri okuma ve ön işleme
- scikit-learn  : Encoding, ölçekleme, model eğitimi ve sınıflandırma metrikleri

Çalıştırma Adımları:
1. Gerekli kütüphaneleri yükleyin: pip install -r requirements.txt
2. Modeli çalıştırın: python3 musteri_churn_tahmini.py
   (musteri_churn_verisi.csv repoda hazır olarak geliyor. Dosya silinirse
   'python3 veri_seti_olustur.py' ile aynı veri tekrar üretilebilir, seed sabit.)
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# 1. DOSYA YOLU VE VERİ SETİ YÜKLEME
KLASOR = os.path.dirname(os.path.abspath(__file__))
CSV_YOLU = os.path.join(KLASOR, "musteri_churn_verisi.csv")

# Eğer veri seti klasörde yoksa uyarı ver
if not os.path.exists(CSV_YOLU):
    raise FileNotFoundError("musteri_churn_verisi.csv bulunamadı! Lütfen önce 'veri_seti_olustur.py' dosyasını çalıştırın.")

df = pd.read_csv(CSV_YOLU)

# 2. TEMEL VERİ İNCELEME
print("="*50)
print("1. TEMEL VERİ İNCELEME")
print("="*50)
print("\n--- İlk 5 Satır ---")
print(df.head())

print(f"\nVeri Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

print("\n--- Hedef Değişken (Churn) Dağılımı ---")
print(df["churn"].value_counts())
print(f"Ayrılan Müşteri Oranı: %{df['churn'].mean()*100:.2f}")

# 3. EKSİK DEĞER KONTROLÜ VE DOLDURMA
print("\n" + "="*50)
print("2. EKSİK DEĞER KONTROLÜ VE DOLDURMA")
print("="*50)
print("\nEksik Değer Sayıları:")
print(df.isnull().sum())

# Gelir için medyan, kategorik alanlar için en sık tekrar eden değer (mode) kullanıyoruz
df["gelir"] = df["gelir"].fillna(df["gelir"].median())
df["uyelik_tipi"] = df["uyelik_tipi"].fillna(df["uyelik_tipi"].mode()[0])

print(f"\nDoldurma sonrası kalan eksik değer sayısı: {df.isnull().sum().sum()}")

# 4. YENİ ÖZNİTELİK ÜRETME (Feature Engineering)
print("\n" + "="*50)
print("3. YENİ ÖZNİTELİK ÜRETME")
print("="*50)

# Ay cinsinden süreyi yıla çeviriyoruz
df["abonelik_yili"] = (df["abonelik_suresi"] / 12).round(2)

# Müşterinin destek talebi açıp açmadığını belirten bayrak (0 veya 1)
df["destek_talebi_var_mi"] = (df["destek_talebi_sayisi"] > 0).astype(int)

# Geliri 3 gruba bölüyoruz
df["gelir_grubu"] = pd.cut(df["gelir"],
                           bins=[0, 25000, 45000, np.inf],
                           labels=["dusuk", "orta", "yuksek"])

print("\nÜretilen Öznitelik Örnekleri:")
print(df[["gelir", "gelir_grubu", "abonelik_suresi", "abonelik_yili", "destek_talebi_var_mi"]].head())

# 5. KATEGORİK DEĞİŞKENLERİ DÖNÜŞTÜRME (One-Hot Encoding)
kategorik_sutunlar = ["sehir", "uyelik_tipi", "gelir_grubu"]
df_encoded = pd.get_dummies(df, columns=kategorik_sutunlar, drop_first=True, dtype=int)

# 6. VERİYİ TRAIN / VALIDATION / TEST KÜMELERİNE AYIRMA
# abonelik_yili, abonelik_suresi'nin 12'ye bölünmüş hali olduğu için aynı bilgiyi
# iki kere vermemek adına modele sokmuyoruz (inceleme amaçlı ürettik).
X = df_encoded.drop(["churn", "abonelik_yili"], axis=1)
y = df_encoded["churn"]

# Önce test setini ayırıyoruz (%20), sonra kalan veriyi train (%60) ve
# validation (%20) olarak bölüyoruz. Stratify ile churn oranını her kümede koruyoruz.
X_gecici, X_test, y_gecici, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_gecici, y_gecici, test_size=0.25, random_state=42, stratify=y_gecici
)

print("\n" + "="*50)
print("4. VERİ BÖLÜMLEME (Train / Validation / Test)")
print("="*50)
print(f"Eğitim Veri Seti Boyutu     : {X_train.shape}")
print(f"Validation Veri Seti Boyutu : {X_val.shape}")
print(f"Test Veri Seti Boyutu       : {X_test.shape}")

# 7. SAYISAL DEĞİŞKENLERİ ÖLÇEKLEME
# Scaler'ı sadece train verisine fit ediyoruz; validation ve test'e yalnızca
# transform uyguluyoruz (veri sızıntısını önlemek için).
sayisal_sutunlar = ["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi"]

scaler = StandardScaler()
X_train_s = X_train.copy()
X_val_s = X_val.copy()
X_test_s = X_test.copy()

X_train_s[sayisal_sutunlar] = scaler.fit_transform(X_train[sayisal_sutunlar])
X_val_s[sayisal_sutunlar] = scaler.transform(X_val[sayisal_sutunlar])
X_test_s[sayisal_sutunlar] = scaler.transform(X_test[sayisal_sutunlar])

# 8. MODELLERİ VALIDATION ÜZERİNDE KARŞILAŞTIRMA
print("\n" + "="*50)
print("5. VALIDATION İLE MODEL KARŞILAŞTIRMA")
print("="*50)

modeller = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42)  # bonus model
}

val_sonuclari = {}

for isim, model in modeller.items():
    model.fit(X_train_s, y_train)
    val_tahmin = model.predict(X_val_s)
    val_acc = accuracy_score(y_val, val_tahmin)
    val_f1 = f1_score(y_val, val_tahmin, zero_division=0)
    val_sonuclari[isim] = val_f1
    print(f"-> {isim:20s} | Validation Accuracy: {val_acc:.3f} | Validation F1: {val_f1:.3f}")

en_iyi_isim = max(val_sonuclari, key=val_sonuclari.get)
en_iyi_model = modeller[en_iyi_isim]
en_iyi_val_f1 = val_sonuclari[en_iyi_isim]

print(f"\nValidation Sonucunda Seçilen Model: {en_iyi_isim}")

# 9. SEÇİLEN MODELİN TEST SETİ ÜZERİNDE DEĞERLENDİRİLMESİ
print("\n" + "="*50)
print(f"6. FİNAL TEST DEĞERLENDİRMESİ ({en_iyi_isim})")
print("="*50)

y_test_pred = en_iyi_model.predict(X_test_s)

acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, zero_division=0)
rec = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)

print("\n--- Confusion Matrix (Karmaşıklık Matrisi) ---")
print(confusion_matrix(y_test, y_test_pred))

print("\n--- Sınıflandırma Metrikleri ---")
print(f"Accuracy  (Doğruluk)   : {acc:.3f}")
print(f"Precision (Kesinlik)   : {prec:.3f}")
print(f"Recall    (Duyarlılık) : {rec:.3f}")
print(f"F1-Score  (F1 Skoru)   : {f1:.3f}")

print("\n--- Detaylı Sınıflandırma Raporu ---")
print(classification_report(y_test, y_test_pred, target_names=["Kalır (0)", "Ayrılır (1)"], zero_division=0))

# 10. SONUÇ VE MODEL YORUMU
print("\n" + "="*50)
print("7. SONUÇ VE MODEL YORUMU")
print("="*50)

# Her model kazanırsa yorumda kullanılacak kısa gerekçeler
gerekceler = {
    "Logistic Regression": "veri seti küçük ve sınıflar arasındaki ilişki görece basit olduğu için doğrusal bir modelin genelleme yapması daha kolay oldu",
    "KNN (k=5)": "ölçekleme sonrası benzer profildeki müşterilerin birbirine yakın düşmesi mesafe tabanlı modele avantaj sağladı",
    "Decision Tree": "churn'ün destek talebi sayısı ve abonelik süresi gibi eşik tabanlı kurallarla ilişkili olması ağaç yapısına avantaj sağladı",
}

siralama = sorted(val_sonuclari.items(), key=lambda x: x[1], reverse=True)
fark = siralama[0][1] - siralama[1][1]

print(f"""
- Validation karşılaştırmasında en iyi model {en_iyi_isim} oldu (Validation F1 = {en_iyi_val_f1:.3f}).
- Bence bunun nedeni: {gerekceler[en_iyi_isim]}.
- Seçilen modelin test sonuçları: Accuracy = {acc:.3f}, Recall = {rec:.3f}, F1 = {f1:.3f}.
- Veri setinde ayrılan müşteri (churn=1) oranı azınlıkta olduğu için accuracy tek başına
  yeterli bir başarı ölçütü değil; bu yüzden model seçiminde F1 skorunu baz aldım.""")

if fark < 0.05:
    print(f"- Not: En iyi iki modelin validation F1 farkı oldukça küçük ({fark:.3f}). Veri seti küçük"
          "\n  olduğu için bu fark kesin bir üstünlük anlamına gelmeyebilir.")
