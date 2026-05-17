import pandas as pd
from sqlalchemy import create_engine
import time

# --- KONFIGURASI ---
# Ganti path ini sesuai lokasi file CSV hasil download dari Kaggle
file_path = 'PS_20174392719_1491204439457_log.csv' 

# Koneksi ke MySQL (XAMPP default: user='root', password kosong)
# Format: mysql+mysqlconnector://user:password@host/db_name
db_connection_str = 'mysql+mysqlconnector://root:@localhost/fraud_project'
db_connection = create_engine(db_connection_str)

print("🚀 Memulai proses migrasi data...")

# --- PROSES CHUNKING ---
# Kita baca per 50.000 baris biar RAM hemat
chunk_size = 50000 
batch_no = 1

try:
    # Membaca CSV secara bertahap (chunk by chunk)
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        
        # 1. RENAME KOLOM
        # Nama kolom di CSV beda sama di SQL yang kita buat tadi, harus disamakan.
        # Format: {'nama_di_csv': 'nama_di_sql'}
        chunk = chunk.rename(columns={
            'step': 'step',
            'type': 'trans_type',          # Di CSV 'type', di SQL 'trans_type'
            'amount': 'amount',
            'nameOrig': 'name_orig',       # CamelCase ke snake_case
            'oldbalanceOrg': 'old_balance_orig',
            'newbalanceOrig': 'new_balance_orig',
            'nameDest': 'name_dest',
            'oldbalanceDest': 'old_balance_dest',
            'newbalanceDest': 'new_balance_dest',
            'isFraud': 'is_fraud',
            'isFlaggedFraud': 'is_flagged_fraud'
        })
        
        # 2. MASUKKAN KE MYSQL dengan insert batch size lebih kecil
        # if_exists='append' artinya nambahin data, bukan hapus tabel
        # chunksize=1000 mencegah "max_allowed_packet" error
        chunk.to_sql('raw_transactions', con=db_connection, if_exists='append', index=False, chunksize=1000)
        
        print(f"✅ Batch {batch_no} berhasil masuk ({chunk_size * batch_no} baris)")
        batch_no += 1
        
        # Opsional: Uncomment baris bawah ini kalau mau test dulu (cuma masukin 2 batch lalu stop)
        # if batch_no > 2: break 

    print("🎉 SELESAI! Semua data berhasil masuk ke MySQL.")

except Exception as e:
    print(f"❌ Terjadi Error: {e}")