# 🚀 Quick Start Guide

## 📁 File yang Diperlukan (Hanya 2 File!)

### 1. Backend API Server

- **File**: `api_final.py` ✅ **GUNAKAN INI**
- **Jalankan**: `python api_final.py`

### 2. Frontend Streamlit App

- **File**: `app_final.py` ✅ **GUNAKAN INI**
- **Jalankan**: `streamlit run app_final.py`

## ❌ File Lain (Abaikan/Hapus)

- `api.ipynb` → Jupyter notebook (old version)
- `app.py` → Streamlit app lama (broken)
- `api_simple.py` → API sederhana (ganti dengan api_final.py)
- `app_fixed.py` → Streamlit app temporer (ganti dengan app_final.py)
- `api_lightweight.py` → API yang gagal dibuat
- `train_model.py` → Training script (jika perlu training)

## 🎯 Cara Run (2 Langkah Saja!)

### Step 1: Install Dependencies

```bash
pip install flask flask-cors requests opencv-python streamlit pillow numpy pandas
```

### Step 2: Jalankan Kedua File

```bash
# Terminal 1 - Jalankan API Server
python api_final.py

# Terminal 2 - Jalankan Streamlit App
streamlit run app_final.py
```

## 🌐 Akses

- **API Server**: http://localhost:5000
- **Streamlit App**: http://localhost:8501

## ✨ Features

- ✅ **API Mode**: Hubungkan ke server deteksi
- ✅ **Demo Mode**: Testing tanpa API server
- ✅ **29 Indonesian Foods**: Nasi goreng, ayam goreng, dll
- ✅ **Nutrition Analysis**: Dengan Groq LLM
- ✅ **Modern UI**: Responsive design

## 📞 Help

- Cek `README_FIX.md` untuk detail lengkap
- Pastikan kedua file berjalan untuk mode API
- Gunakan Demo Mode untuk testing cepat
