import numpy as np
import pandas as pd

# Test seti için farklı bir başlangıç noktası (seed) seçiyoruz
np.random.seed(101)

# Oranlar tersine döndü: 1500 verinin %20'si Patojenik, %80'i Benign
patojenik_sayisi_test = 300
benign_sayisi_test = 1200

print("1. Test seti için Patojenik veriler üretiliyor...")
maf_p = np.random.exponential(scale=0.001, size=patojenik_sayisi_test)
cadd_p = np.random.normal(loc=25, scale=5, size=patojenik_sayisi_test)
pol_p = np.random.normal(loc=2.5, scale=0.5, size=patojenik_sayisi_test) 
hyd_p = np.random.normal(loc=2.5, scale=0.5, size=patojenik_sayisi_test)
ekstra1_p = np.random.normal(0, 1, size=patojenik_sayisi_test)
ekstra2_p = np.random.normal(0, 1, size=patojenik_sayisi_test)

df_patojenik_test = pd.DataFrame({
    'feat_0': maf_p, 'feat_1': cadd_p, 'feat_2': pol_p, 
    'feat_3': hyd_p, 'feat_4': ekstra1_p, 'feat_5': ekstra2_p
})
etiket_patojenik_test = np.ones(patojenik_sayisi_test, dtype=int)


print("2. Test seti için Benign veriler üretiliyor...")
maf_b = np.random.beta(a=2, b=8, size=benign_sayisi_test)
cadd_b = np.random.normal(loc=10, scale=6, size=benign_sayisi_test)
pol_b = np.random.normal(loc=0.5, scale=0.2, size=benign_sayisi_test)
hyd_b = np.random.normal(loc=0.5, scale=0.2, size=benign_sayisi_test)
ekstra1_b = np.random.normal(0, 1, size=benign_sayisi_test)
ekstra2_b = np.random.normal(0, 1, size=benign_sayisi_test)

df_benign_test = pd.DataFrame({
    'feat_0': maf_b, 'feat_1': cadd_b, 'feat_2': pol_b, 
    'feat_3': hyd_b, 'feat_4': ekstra1_b, 'feat_5': ekstra2_b
})
etiket_benign_test = np.zeros(benign_sayisi_test, dtype=int)


print("3. Veriler birleştirilip karıştırılıyor...")
df_tamami_test = pd.concat([df_patojenik_test, df_benign_test], ignore_index=True)
etiketler_tamami_test = np.concatenate([etiket_patojenik_test, etiket_benign_test])

df_tamami_test['etiket'] = etiketler_tamami_test
df_tamami_test = df_tamami_test.sample(frac=1, random_state=101).reset_index(drop=True)

cevap_anahtari_test = df_tamami_test[['etiket']]
test_ozellikleri = df_tamami_test.drop(columns=['etiket'])


print("4. Dosyalar kaydediliyor...")
# Bu dosyayı model ekibine VERECEKSİN
test_ozellikleri.to_parquet('test_seti_ozellikler.parquet')

# BU DOSYAYI KESİNLİKLE KİMSEYE VERMİYORSUN, SENDE KALACAK
cevap_anahtari_test.to_csv('test_seti_etiketler_GIZLI.csv', index=False)

print("\n✅ GÖREV 3 TAMAMLANDI!")
print("Model grubuna 'test_seti_ozellikler.parquet' dosyasını iletebilirsin.")