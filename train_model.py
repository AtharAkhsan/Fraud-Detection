import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # Ini Backpropagation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib # Untuk nge-save model

# 1. LOAD DATA
print("📂 Loading data hasil olahan...")
df = pd.read_csv('processed_fraud_data.csv')

# 2. SPLIT DATA
# X = Semua kolom KECUALI 'is_fraud'
X = df.drop('is_fraud', axis=1)
# y = Kolom 'is_fraud' saja
y = df['is_fraud']

# Bagi jadi Training (80%) dan Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Data Split: {X_train.shape[0]} training, {X_test.shape[0]} testing")

# 3. SCALING (PENTING BUAT NEURAL NETWORK)
# Neural Network sensitif sama angka besar/kecil, jadi kita standarisasi dulu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAINING MODEL (BACKPROPAGATION)
print("🧠 Sedang melatih Neural Network (Backpropagation)...")
print("   (Bisa memakan waktu 1-2 menit tergantung spek laptop)")

# MLP = Multi Layer Perceptron (Basisnya Backpropagation)
# hidden_layer_sizes=(64, 32) -> Ada 2 layer tersembunyi, layer 1 punya 64 neuron, layer 2 punya 32 neuron.
model = MLPClassifier(hidden_layer_sizes=(64, 32), 
                      activation='relu', 
                      solver='adam', 
                      max_iter=500, 
                      random_state=42)

model.fit(X_train_scaled, y_train)

# 5. EVALUASI
print("\n📝 HASIL EVALUASI MODEL:")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Tampilkan Confusion Matrix sederhana
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"✅ Benar Legit (True Negative): {tn}")
print(f"⚠️ Salah Deteksi (False Positive - Disangka Fraud padahal bukan): {fp}")
print(f"❌ Lolos Deteksi (False Negative - Fraud dikira aman): {fn} -> INI YANG BAHAYA")
print(f"🎯 Benar Fraud (True Positive): {tp}")

# 6. SAVE MODEL (SAVE GAME)
# Kita simpan Model dan Scaler-nya biar bisa dipakai di Streamlit
joblib.dump(model, 'model_fraud.pkl')
joblib.dump(scaler, 'scaler_fraud.pkl')

print("\n💾 Model berhasil disimpan sebagai 'model_fraud.pkl'!")