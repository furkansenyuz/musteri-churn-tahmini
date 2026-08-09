import os
import pandas as pd
import numpy as np

# Rastgele veri üretimi için sabit tohum
np.random.seed(42)

n_samples = 300

data = {
    "yas": np.random.randint(18, 65, size=n_samples),
    "gelir": np.random.choice([np.nan, 20000, 32000, 48000, 65000, 85000], size=n_samples, p=[0.05, 0.2, 0.3, 0.25, 0.1, 0.1]),
    "abonelik_suresi": np.random.randint(1, 60, size=n_samples), # ay cinsinden
    "destek_talebi_sayisi": np.random.poisson(lam=2, size=n_samples),
    "sehir": np.random.choice(["Istanbul", "Ankara", "Izmir", "Bursa"], size=n_samples),
    "uyelik_tipi": np.random.choice(["Bronz", "Gumus", "Altin"], size=n_samples, p=[0.50, 0.35, 0.15])
}

df = pd.DataFrame(data)

# uyelik_tipi için eksik değerleri sonradan maske ile ekliyoruz.
# Not: np.random.choice içine np.nan ile string birlikte verilince numpy hepsini
# string'e çevirip nan'ı "nan" yazısına dönüştürüyor, o yüzden bu yöntem daha güvenli.
eksik_maske = np.random.rand(n_samples) < 0.05
df.loc[eksik_maske, "uyelik_tipi"] = np.nan

# Churn olasılığını gerçekçi kurallara bağlayalım:
# Düşük abonelik süresi ve yüksek destek talebi churn riskini artırır,
# üyelik tipi de etkili olsun (Bronz üyeler daha riskli, Altin üyeler daha sadık)
uyelik_etkisi = df["uyelik_tipi"].map({"Bronz": 0.10, "Gumus": 0.0, "Altin": -0.10}).fillna(0.05)
churn_prob = 0.15 + (df["destek_talebi_sayisi"] * 0.12) - (df["abonelik_suresi"] * 0.006) + uyelik_etkisi
churn_prob = np.clip(churn_prob, 0.05, 0.95)

df["churn"] = (np.random.rand(n_samples) < churn_prob).astype(int)

# CSV olarak kaydet
KLASOR = os.path.dirname(os.path.abspath(__file__))
CSV_YOLU = os.path.join(KLASOR, "musteri_churn_verisi.csv")

df.to_csv(CSV_YOLU, index=False)
print(f"'{CSV_YOLU}' başarıyla oluşturuldu! (Toplam {n_samples} satır)")