import numpy as np
import pandas as pd

# Yönergedeki kural: "numpy.random seed=42" (Her çalıştırdığımızda aynı rastgele sayıları versin diye)
np.random.seed(42)

patojenik_sayisi = 1200
benign_sayisi = 300

print("1. Sentetik Patojenik (Hastalık Yapan) veriler üretiliyor...")
# Patojenik MAF: Exponential(0.001) - Nadir görülür
maf_p = np.random.exponential(scale=0.001, size=patojenik_sayisi)
# Patojenik CADD: Normal(mu=25, sigma=5)
cadd_p = np.random.normal(loc=25, scale=5, size=patojenik_sayisi)
# Patojenik amino asit değişimi: Fark büyük olsun (|fark| > 1.5)
pol_p = np.random.normal(loc=2.5, scale=0.5, size=patojenik_sayisi) 
hyd_p = np.random.normal(loc=2.5, scale=0.5, size=patojenik_sayisi)
# Kalan 2 özellik için (toplam 6 özellik istendiği için) rastgele sayılar
ekstra1_p = np.random.normal(0, 1, size=patojenik_sayisi)
ekstra2_p = np.random.normal(0, 1, size=patojenik_sayisi)

# Patojenik verileri tablo yap
df_patojenik = pd.DataFrame({
    'feat_0': maf_p, 'feat_1': cadd_p, 'feat_2': pol_p, 
    'feat_3': hyd_p, 'feat_4': ekstra1_p, 'feat_5': ekstra2_p
})
# Cevap anahtarı: 1 = Patojenik
etiket_patojenik = np.ones(patojenik_sayisi, dtype=int)


print("2. Sentetik Benign (Zararsız) veriler üretiliyor...")
# Benign MAF: Beta(2,8) - Sık görülür
maf_b = np.random.beta(a=2, b=8, size=benign_sayisi)
# Benign CADD: Normal(mu=10, sigma=6)
cadd_b = np.random.normal(loc=10, scale=6, size=benign_sayisi)
# Benign amino asit değişimi: Fark küçük olsun
pol_b = np.random.normal(loc=0.5, scale=0.2, size=benign_sayisi)
hyd_b = np.random.normal(loc=0.5, scale=0.2, size=benign_sayisi)
# Kalan 2 özellik
ekstra1_b = np.random.normal(0, 1, size=benign_sayisi)
ekstra2_b = np.random.normal(0, 1, size=benign_sayisi)

# Benign verileri tablo yap
df_benign = pd.DataFrame({
    'feat_0': maf_b, 'feat_1': cadd_b, 'feat_2': pol_b, 
    'feat_3': hyd_b, 'feat_4': ekstra1_b, 'feat_5': ekstra2_b
})
# Cevap anahtarı: 0 = Benign
etiket_benign = np.zeros(benign_sayisi, dtype=int)


print("3. Veriler birleştirilip karıştırılıyor (Shuffle)...")
# İki tabloyu alt alta ekle
df_tamami = pd.concat([df_patojenik, df_benign], ignore_index=True)
etiketler_tamami = np.concatenate([etiket_patojenik, etiket_benign])

# Karıştırırken etiketler (cevaplar) kaybolmasın diye geçici olarak tabloya ekliyoruz
df_tamami['etiket'] = etiketler_tamami
# Satırları tamamen rastgele karıştır
df_tamami = df_tamami.sample(frac=1, random_state=42).reset_index(drop=True)

# Karıştıktan sonra yönergedeki "Etiket Gizliliği" kuralına göre cevap anahtarını tablodan ayırıyoruz
cevap_anahtari = df_tamami[['etiket']]
egitim_ozellikleri = df_tamami.drop(columns=['etiket'])


print("4. Dosyalar kaydediliyor...")
# Özellikleri parquet formatında kaydet (Model ekibine gönderilecek olan dosya bu)
egitim_ozellikleri.to_parquet('egitim_seti_ozellikler.parquet')
# Cevap anahtarını CSV olarak kaydet (Model ekibi tahmin yaptıktan sonra karşılaştırmak için)
cevap_anahtari.to_csv('egitim_seti_etiketler.csv', index=False)

print("\n✅ GÖREV 2 TAMAMLANDI!")
print("Klasöründe 'egitim_seti_ozellikler.parquet' dosyası oluştu.")