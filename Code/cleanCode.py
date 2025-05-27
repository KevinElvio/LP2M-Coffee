import pandas as pd
import os

# Baca file CSV
df = pd.read_csv(r"d:\KULIAH\AKU CINTA NGODING\LP2M\Code\dataSetKopi_Dirty.csv")

# 1. Hapus kolom yang tidak dibutuhkan
columns_to_drop = ['Unnamed: 0', 'Color.1'] if 'Color.1' in df.columns else ['Unnamed: 0']
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

# 2. Hapus baris yang kosong semua
df_cleaned = df_cleaned.dropna(how='all')

# 3. Isi missing value di kolom numerik dengan nilai rata-rata
numeric_columns = df_cleaned.select_dtypes(include='number').columns
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())

# 4. Isi missing value di kolom teks dengan nilai paling sering (modus)
object_columns = df_cleaned.select_dtypes(include='object').columns
for col in object_columns:
    if df_cleaned[col].isnull().any():
        mode = df_cleaned[col].mode()
        if not mode.empty:
            df_cleaned[col] = df_cleaned[col].fillna(mode[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")

# 5. Reset index
df_cleaned = df_cleaned.reset_index(drop=True)

# 6. Simpan ke file baru
df_cleaned.to_csv("dataSetKopi_Bersih.csv", index=False)

print("Data berhasil dibersihkan dan disimpan sebagai 'dataSetKopi_Bersih.csv'")
