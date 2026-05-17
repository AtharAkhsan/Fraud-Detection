# 🛡️ Corporate Financial Risk & Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![SQL](https://img.shields.io/badge/Database-MySQL-4479A1)
![Machine Learning](https://img.shields.io/badge/Model-Backpropagation%20NN-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **"Simulating the future of Forensic Technology: Detecting high-risk financial anomalies using Neural Networks and Real-time Auditing Dashboards."**

---

## Project Overview

Dalam era transaksi digital yang masif, metode audit tradisional berbasis aturan (*rule-based*) sering kali gagal mendeteksi pola pencucian uang (*money laundering*) yang kompleks. Proyek ini bertujuan untuk membangun sistem **end-to-end Forensic Analytics** yang mampu mengidentifikasi transaksi curang (*fraud*) secara otomatis.

Sistem ini mensimulasikan alur kerja **Risk Assurance** modern, mulai dari penyimpanan data transaksi skala besar di MySQL, pemodelan perilaku menggunakan Deep Learning (Backpropagation), hingga visualisasi dampak kerugian finansial bagi stakeholder non-teknis.

### Key Business Metrics (Impact):
* **Detection Rate (Recall):** 99% (Berhasil mendeteksi mayoritas transaksi fraud).
* **Potential Loss Prevented:** >$631 Million (Simulasi pada dataset sample).
* **Processing Speed:** Real-time anomaly scoring.

---

## Technical Architecture

Proyek ini menggunakan pendekatan **ELT (Extract, Load, Transform)** dan **Machine Learning Pipeline**:

1.  **Data Ingestion (SQL):** Mengelola dataset transaksi (PaySim) menggunakan MySQL untuk simulasi data warehousing yang terstruktur.
2.  **Forensic Feature Engineering:**
    * Mendeteksi pola *Account Emptying* (Saldo awal = Jumlah transfer).
    * Menganalisis *Balance Error* (Ketidaksesuaian logika matematis antara saldo sebelum dan sesudah transaksi).
3.  **AI Modeling (Scikit-Learn):**
    * Algoritma: **Multi-Layer Perceptron (Backpropagation)**.
    * Struktur: Input Layer -> Hidden Layers (Relu) -> Output (Sigmoid).
    * Alasan Pemilihan: Mampu menangkap hubungan non-linear yang kompleks antar variabel transaksi.
4.  **Visualization (Streamlit):** Dashboard interaktif untuk tim audit memantau anomali prioritas tinggi.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Database:** MySQL (SQLAlchemy Connector)
* **Machine Learning:** Scikit-Learn (MLPClassifier), Pandas, NumPy
* **Visualization:** Streamlit, Plotly Express
* **Environment:** VS Code, XAMPP

---

## 🚀 How to Run This Project

Ikuti langkah berikut untuk menjalankan sistem di komputer lokal Anda:

### Prerequisites
* Python 3.x Installed
* MySQL Running (via XAMPP or Workbench)

### Installation
1.  **Clone Repository**
    ```bash
    git clone https://github.com/AtharAkhsan/fraud-detection.git
    cd fraud-detection
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Database**
    * Buat database baru di MySQL bernama `fraud_project`.
    * Import/Jalankan script `ingest_data.py` untuk memasukkan dataset.

4.  **Run Dashboard**
    ```bash
    streamlit run dashboard.py
    ```

---

## 📊 Model Performance Evaluation

Fokus utama dalam deteksi fraud adalah **Recall** (meminimalkan False Negative / fraud yang lolos).

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Legit (0)** | 1.00 | 1.00 | 1.00 |
| **Fraud (1)** | 1.00 | **0.99** | 0.99 |

*Model berhasil menangkap 99% dari seluruh aktivitas mencurigakan dalam data pengujian.*

---

## Author

**Athar Iftikhar Akhsan**
* Student at Faculty of Computer Science, Universitas Brawijaya
* Interest: Data Science, Forensic Tech, & Web Development
* [LinkedIn Profile](www.linkedin.com/in/athar-iftikhar-akhsan) | [GitHub](https://github.com/AtharAkhsan/)

---

*Disclaimer: Dataset used is PaySim (Synthetic Financial Dataset for Fraud Detection) for educational and simulation purposes only.*
