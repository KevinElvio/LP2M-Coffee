import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load data bersih
df = pd.read_csv(r"d:\KULIAH\AKU CINTA NGODING\LP2M\Code\dataSetKopi_Bersih.csv")  # ganti sesuai nama file bersih kamu

# 2. Lihat kolom-kolom deskriptif
print(df.columns)  # lihat apakah ada kolom seperti 'deskripsi', 'rasa', 'aroma', dll.

# 3. Gabungkan kolom menjadi satu kolom teks jika perlu
df["full_text"] = df[["Processing.Method", "Region", "Variety"]].fillna('').astype(str).agg(' '.join, axis=1)

# 4. Buat vektor TF-IDF dari deskripsi kopi
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["full_text"])

# 5. Fungsi untuk mencari kopi berdasarkan input deskripsi
def search_kopi(input_text, top_n=3):
    input_vec = vectorizer.transform([input_text])
    similarity = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[::-1][:top_n]
    hasil = df.iloc[top_indices].copy()

    # Tambahkan kolom similarity agar tahu seberapa mirip
    hasil["Similarity"] = similarity[top_indices]

    # Pilih kolom yang ingin ditampilkan
    kolom_tampil = ["Species", "Processing.Method", "Region", "Variety", "Similarity"]
    return hasil[kolom_tampil].reset_index(drop=True)



# 6. Contoh pemakaian
hasil = search_kopi("Saya memiliki kopi yang wangi dan berwarna hijau serta kopi saya berasal dari negara eropa", top_n=5)
print(hasil.to_string(index=False))

for i, row in hasil.iterrows():
    print(f"\nRekomendasi #{i+1}")
    print(f"Species        : {row['Species']}")
    print(f"Processing     : {row['Processing.Method']}")
    print(f"Region         : {row['Region']}")
    print(f"Variety        : {row['Variety']}")
    print(f"Similarity     : {row['Similarity']:.4f}")

