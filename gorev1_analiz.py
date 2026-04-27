import pandas as pd

# Dosya adını buraya yazıyoruz (Python dosyasıyla aynı klasörde olduklarından emin ol)
dosya_yolu = "variant_summary.txt.gz"

print("1. ClinVar verisi okunuyor... (Bu işlem bilgisayarının hızına göre biraz sürebilir)")
# low_memory=False yapıyoruz çünkü dosya çok büyük, bellek uyarısı vermesin.
df = pd.read_csv(dosya_yolu, sep='\t', compression='gzip', low_memory=False)

print(f"-> Başlangıçta toplam {len(df)} adet varyant var.\n")

print("2. 'reviewed by expert panel' (Uzman Onaylı) filtresi uygulanıyor...")
guvenilir_veri = df[df['ReviewStatus'] == "reviewed by expert panel"]

print(f"-> Uzman onaylı varyant sayısı: {len(guvenilir_veri)}\n")

print("3. Patojenik (Hasta) ve Benign (Sağlıklı) sınıfları ayrıştırılıyor...")
patojenik_olanlar = guvenilir_veri[guvenilir_veri['ClinicalSignificance'] == 'Pathogenic']
benign_olanlar = guvenilir_veri[guvenilir_veri['ClinicalSignificance'] == 'Benign']

print("-" * 40)
print(f"SONUÇLAR:")
print(f"🎯 Gerçek Patojenik Varyant Sayısı: {len(patojenik_olanlar)}")
print(f"🟢 Gerçek Benign Varyant Sayısı: {len(benign_olanlar)}")
print("-" * 40)