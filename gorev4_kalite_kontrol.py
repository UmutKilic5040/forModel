import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

print("1. Ürettiğimiz Eğitim Seti Yükleniyor...")
ozellikler = pd.read_parquet('egitim_seti_ozellikler.parquet')
etiketler = pd.read_csv('egitim_seti_etiketler.csv')

# Analiz kolaylığı için ikisini yan yana birleştiriyoruz
df = pd.concat([ozellikler, etiketler], axis=1)

print("\n2. Temel Sağlık Kontrolleri (Panel Boyutları ve Eksik Değerler):")
print(f"-> Veri Boyutu: {df.shape} (Beklenen: 1500 satır, 7 kolon)")
eksik_veri_sayisi = df.isnull().sum().sum()
print(f"-> Eksik (Null) Değer Sayısı: {eksik_veri_sayisi} (Beklenen: 0)")

if df.shape[0] == 1500 and eksik_veri_sayisi == 0:
    print("   ✅ Veri bütünlüğü kusursuz!")

print("\n3. İstatistiksel Ayrışma Gücü (Mann-Whitney U Testi):")
for kolon in ozellikler.columns:
    hasta_degerleri = df[df['etiket'] == 1][kolon]
    saglikli_degerleri = df[df['etiket'] == 0][kolon]
    
    stat, p_degeri = mannwhitneyu(hasta_degerleri, saglikli_degerleri)
    durum = "Mükemmel Ayrışma!" if p_degeri < 0.05 else "Zayıf Ayrışma"
    print(f"-> {kolon}: p-değeri = {p_degeri:.2e} ({durum})")

print("\n4. Grafik Çiziliyor (feat_1 / CADD Skoru Örneği)...")
plt.figure(figsize=(8, 5))
sns.violinplot(x='etiket', y='feat_1', data=df, palette='muted', inner="quartile")
plt.title('CADD Skoru (feat_1) Dağılımı: Benign (0) vs Patojenik (1)')
plt.savefig('kalite_grafigi_feat1.png')
print("   ✅ Grafik 'kalite_grafigi_feat1.png' olarak kaydedildi.")

print("\n5. Pandas ile Temel HTML Raporu Oluşturuluyor...")
# Pandas'ın istatistik özetini doğrudan HTML tablosuna çeviriyoruz
html_rapor = df.groupby('etiket').describe().T.to_html()
with open("basit_kalite_raporu.html", "w", encoding="utf-8") as f:
    f.write(f"<h1>Grup B-1 Sentetik Veri Kalite Raporu</h1><br>{html_rapor}")
print("   ✅ HTML raporu 'basit_kalite_raporu.html' olarak kaydedildi.")

print("\n🎉 TÜM GÖREVLER TAMAMLANDI!")