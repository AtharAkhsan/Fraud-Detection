import pandas as pd
from sqlalchemy import create_engine
import numpy as np

# 1. KONEKSI KE DATABASE
db_connection_str = 'mysql+mysqlconnector://root:@localhost/fraud_project'
db_connection = create_engine(db_connection_str)

print("⏳ Sedang mengambil data dari MySQL...")

# 2. SMART DATA LOADING (Undersampling)
# Kita ambil semua yang Fraud (penting!), tapi batasi yang Legit biar dataset seimbang
query_fraud = "SELECT * FROM raw_transactions WHERE is_fraud = 1"
query_legit = "SELECT * FROM raw_transactions WHERE is_fraud = 0 LIMIT 50000"

df_fraud = pd.read_sql(query_fraud, db_connection)
df_legit = pd.read_sql(query_legit, db_connection)

# Gabungkan
df = pd.concat([df_fraud, df_legit], axis=0).reset_index(drop=True)

print(f"✅ Data berhasil diload! Total baris: {len(df)}")
print(f"   - Fraud: {len(df_fraud)}")
print(f"   - Legit: {len(df_legit)}")

# 3. FEATURE ENGINEERING (Bagian "Forensic" PwC)
print("⚙️ Melakukan Feature Engineering...")

# A. Error Balance Origin (Selisih saldo pengirim yang aneh)
# Logika: Saldo Awal - Jumlah tf = Saldo Akhir. Kalau beda, berarti sistem error atau dimanipulasi.
df['error_bal_orig'] = df['new_balance_orig'] + df['amount'] - df['old_balance_orig']

# B. Error Balance Dest (Selisih saldo penerima yang aneh)
# Logika: Saldo Awal Penerima + Jumlah tf = Saldo Akhir Penerima.
df['error_bal_dest'] = df['old_balance_dest'] + df['amount'] - df['new_balance_dest']

# C. One-Hot Encoding untuk Tipe Transaksi
# Mengubah 'CASH_OUT', 'TRANSFER' jadi angka (0, 1) biar bisa dibaca AI
df = pd.get_dummies(df, columns=['trans_type'], drop_first=True)

# 4. BERSIH-BERSIH (Data Cleaning)
# Hapus kolom yang tidak relevan buat AI (Nama orang/string tidak bisa dihitung)
cols_to_drop = ['name_orig', 'name_dest', 'is_flagged_fraud', 'created_at']
df_clean = df.drop(columns=cols_to_drop)

# 5. SIMPAN HASILNYA
output_file = 'processed_fraud_data.csv'
df_clean.to_csv(output_file, index=False)

print(f"🎉 Selesai! Data siap pakai disimpan sebagai '{output_file}'")
print("Preview 5 baris data hasil olahan:")
print(df_clean[['amount', 'error_bal_orig', 'is_fraud']].head())