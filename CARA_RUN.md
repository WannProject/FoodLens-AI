# 🚀 Cara Run Sistem Deteksi Makanan

## 📁 File yang Diperlukan (Hanya 2 File!)

### ✅ **Gunakan file ini:**

1. **Backend**: `api.py` (API Server dengan simulasi)
2. **Frontend**: `app.py` (Streamlit app dengan mode demo)

### ❌ **File lain (abaikan):**

- `api_final.py`, `app_final.py` - Versi lain yang lebih kompleks
- `api_simple.py`, `app_fixed.py` - Versi temporary
- `train_model.py` - Untuk training (jika perlu)
- File `.md` lainnya - Dokumentasi

## 🔧 Cara Menjalankan (3 Langkah)

### Step 1: Install Dependencies

```bash
pip install flask flask-cors requests opencv-python streamlit pillow numpy pandas
```

### Step 2: Jalankan Backend API

```bash
python api.py
```

Akan muncul:

```
🍽️ Food Detection API Server Starting...
📍 Server: http://localhost:5000
🎭 Mode: Simulation (Demo)
```

### Step 3: Jalankan Frontend Streamlit

```bash
streamlit run app.py
```

Akan otomatis buka browser di: http://localhost:8501

## 🎮 Cara Pakai

### di Streamlit App (http://localhost:8501):

1. **Pilih Mode**: "Demo Mode" (rekomendasi) atau "Local (Built-in)"
2. **Upload Gambar**: Pilih foto makanan (JPG/PNG)
3. **Klik "🔎 Deteksi Gizi"**
4. **Lihat Hasil**:
   - Gambar dengan bounding box
   - Tabel deteksi makanan
   - Analisis gizi dari Groq API

### Test API Server:

```bash
curl http://localhost:5000/health
```

## 🚨 Jika Ada Error

### 1. "ModuleNotFoundError"

```bash
pip install flask flask-cors requests opencv-python streamlit pillow numpy pandas
```

### 2. "Cannot connect to API server"

- Pastikan `python api.py` sudah berjalan di terminal lain
- Cek port 5000 tidak digunakan aplikasi lain

### 3. "Groq API key not configured"

- Di sidebar Streamlit, masukkan API key dari https://console.groq.com/
- Atau gunakan Demo Mode (tidak perlu API key)

## ✨ Features yang Berfungsi

- ✅ **Deteksi 29 makanan Indonesia** (simulasi)
- ✅ **Bounding box visualization**
- ✅ **Analisis gizi dengan Groq LLM**
- ✅ **Confidence score filtering**
- ✅ **Download hasil gambar**
- ✅ **Modern responsive UI**

## 📞 Contoh Hasil

```
🎭 Demo Mode - Simulating detection results...
✅ Detection Berhasil! Ditemukan 2 objek makanan: nasi goreng, ayam goreng

| Makanan | Kalori | Protein | Karbohidrat | Lemak |
|---------|--------|---------|-------------|-------|
| Nasi Goreng | 250 kcal | 5g | 45g | 8g |
| Ayam Goreng | 280 kcal | 25g | 0g | 18g |
```

## 🎯 Tips

- **Demo Mode**: Untuk testing cepat tanpa API
- **API Mode**: Untuk hasil real-time dengan backend
- **Confidence Threshold**: Filter deteksi dengan score minimal
- **Groq API**: Diperlukan untuk analisis gizi yang lebih baik

---

**Status**: ✅ Ready to Use!  
**Files Needed**: `api.py` + `app.py`  
**Ports**: 5000 (API) + 8501 (Streamlit)
