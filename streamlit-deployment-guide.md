# 🚀 Streamlit Cloud Deployment Guide

## ❌ Masalah Utama: Flask API tidak bisa di-deploy ke Streamlit Cloud

**Kenapa Flask API tidak jalan di Streamlit Cloud?**

- Streamlit Cloud hanya menjalankan **Streamlit apps**, bukan Flask server
- Tidak bisa menjalankan multiple services (Flask + Streamlit)
- Port management tidak diizinkan di Streamlit Cloud
- Background process tidak supported

## ✅ Solusi: Streamlit-Native Implementation

### File yang Digunakan untuk Streamlit Cloud

- 🎯 **Main File**: [`streamlit-compatible-api.py`](streamlit-compatible-api.py:1)
- 📦 **Dependencies**: [`requirements.txt`](requirements.txt:1) (updated)
- 🤖 **Model**: `runs/detect/train2/weights/best.pt`

## 📋 Perbandingan Architecture

### ❌ Local Development (Flask + Streamlit)

```
┌─────────────────┐    ┌─────────────────┐
│   Flask API     │    │   Streamlit     │
│   Port 5000     │◄──►│   Port 8501     │
│   YOLO + LLM    │    │   Frontend      │
└─────────────────┘    └─────────────────┘
```

### ✅ Streamlit Cloud (Single App)

```
┌─────────────────────────────────────┐
│         Streamlit App              │
│  ┌─────────────────────────────┐   │
│  │    YOLO Detection           │   │
│  │    Groq LLM Analysis        │   │
│  │    Streamlit UI             │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## 🚀 Cara Deploy ke Streamlit Cloud

### Step 1: Prepare Repository Structure

```
food-detection/
├── streamlit-compatible-api.py    # Main app file
├── requirements.txt               # Dependencies
├── runs/
│   └── detect/
│       └── train2/
│           └── weights/
│               └── best.pt       # YOLO model
└── README.md                     # Documentation
```

### Step 2: Update requirements.txt

```txt
# Core ML/DL Libraries
torch==2.9.0
torchvision==0.24.0
ultralytics==8.3.225

# Image Processing
opencv-python==4.12.0.88
pillow==12.0.0

# Data Processing
numpy==2.2.6

# Web Framework & Frontend
streamlit==1.40.0

# HTTP Requests
requests==2.32.5

# API Integration
groq==0.33.0

# Utilities
PyYAML==6.0.3
```

### Step 3: Upload ke GitHub

1. Push semua file ke GitHub repository
2. Pastikan `best.pt` model termasuk (atau gunakan Git LFS)
3. Commit dan push ke main branch

### Step 4: Deploy ke Streamlit Cloud

1. Login ke [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect GitHub repository
4. Select main branch
5. Main file path: `streamlit-compatible-api.py`
6. Click "Deploy"

## 🔧 Konfigurasi Environment Variables

### Method 1: Streamlit Cloud Secrets

Di Streamlit Cloud dashboard:

1. Go to app settings
2. Advanced → Secrets
3. Add TOML configuration:

```toml
[groq]
api_key = "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"
model = "meta-llama/llama-4-maverick-17b-128e-instruct"
```

### Method 2: Update Code untuk Environment Variables

```python
import streamlit as st
import os

# Use secrets in production, fallback to hardcoded for development
try:
    GROQ_API_KEY = st.secrets["groq"]["api_key"]
    GROQ_MODEL = st.secrets["groq"]["model"]
except:
    GROQ_API_KEY = "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"
    GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
```

## 📱 Features Streamlit-Compatible App

### ✅ UI Components

- Image upload dengan drag & drop
- Real-time detection results
- Confidence threshold slider
- Nutritional analysis display
- Export functionality

### ✅ Backend Integration

- YOLO model loading dengan `@st.cache_resource`
- Groq API integration
- Error handling dan validation
- Base64 image encoding

### ✅ Performance Optimizations

- Model caching untuk fast loading
- Efficient image processing
- Memory management
- Timeout handling

## 🧪 Testing Local Sebelum Deploy

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Local Streamlit

```bash
streamlit run streamlit-compatible-api.py
```

### Test Features

1. Upload image makanan
2. Check detection results
3. Verify nutritional analysis
4. Test export functionality

## 🔍 Troubleshooting Streamlit Cloud

### Common Issues & Solutions

#### 1. Model Loading Error

**Problem**: `FileNotFoundError: runs/detect/train2/weights/best.pt`
**Solution**:

- Pastikan model file ada di repository
- Gunakan Git LFS untuk large files
- Check path configuration

#### 2. Memory Issues

**Problem**: App crashes during model loading
**Solution**:

- Use `@st.cache_resource` for model
- Optimize model size
- Monitor resource usage

#### 3. Groq API Timeout

**Problem**: `requests.exceptions.Timeout`
**Solution**:

- Increase timeout value
- Add retry mechanism
- Check API key validity

#### 4. Dependencies Conflicts

**Problem**: `ImportError` atau version conflicts
**Solution**:

- Use specific versions in requirements.txt
- Remove unused dependencies
- Test with fresh environment

## 📊 Resource Requirements

### Minimum Requirements

- **RAM**: 2GB (recommended: 4GB)
- **CPU**: 2 cores (recommended: 4 cores)
- **Storage**: 500MB untuk model + app

### Optimization Tips

```python
# Use caching for expensive operations
@st.cache_resource
def load_model():
    return YOLO(model_path)

# Limit concurrent requests
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = True
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Test and Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Test imports
        run: |
          python -c "import streamlit, ultralytics, cv2, numpy; print('Success')"
```

## 📈 Monitoring & Analytics

### Streamlit Cloud Metrics

- App performance dashboard
- Error tracking
- Usage statistics
- Resource monitoring

### Custom Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log detection events
logger.info(f"Detected {len(detected_objects)} objects")
```

## 🎯 Best Practices

### 1. Security

- Use environment variables for API keys
- Validate user inputs
- Rate limiting for API calls

### 2. Performance

- Cache expensive operations
- Optimize image sizes
- Use lazy loading

### 3. User Experience

- Progress indicators
- Error messages
- Responsive design

### 4. Maintenance

- Regular dependency updates
- Model versioning
- Backup strategies

## 🚀 Deployment Checklist

- [ ] Repository structure correct
- [ ] requirements.txt updated
- [ ] Model files included
- [ ] Environment variables configured
- [ ] Local testing successful
- [ ] GitHub repository synced
- [ ] Streamlit Cloud configured
- [ ] App deployed and tested
- [ ] Monitoring set up

## 📞 Support

**Resources**:

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

**Common Commands**:

```bash
# Local development
streamlit run streamlit-compatible-api.py

# Check dependencies
pip install -r requirements.txt

# Test model loading
python -c "from ultralytics import YOLO; print('YOLO imported')"
```

---

**🎉 Selamat! App Anda siap di-deploy ke Streamlit Cloud!**

Gunakan `streamlit-compatible-api.py` sebagai main file untuk deployment yang sukses di Streamlit Cloud.
