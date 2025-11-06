# Quick Commands - Langsung Jalankan

## Cara Cepat Buat Requirements File

### Step 1: Buka Terminal/CMD di Project Folder

```bash
cd d:/MAGANG-DIGIDES/Streamlit/DataLens-AI/food-detection
```

### Step 2: Cek Installed Packages

```bash
# List semua packages
pip list

# Atau format freeze
pip freeze
```

### Step 3: Generate Requirements File

```bash
# Method 1: Generate semua packages
pip freeze > requirements.txt

# Method 2: Generate hanya packages penting (Windows)
pip freeze | findstr /i "ultralytics flask opencv numpy streamlit torch requests jupyter pillow" > requirements.txt

# Method 3: Generate hanya packages penting (Linux/Mac)
pip freeze | grep -E "(ultralytics|flask|opencv|numpy|streamlit|torch|requests|jupyter|pillow)" > requirements.txt
```

### Step 4: Lihat Hasil Requirements File

```bash
# Buka dengan notepad (Windows)
notepad requirements.txt

# Atau lihat di terminal
type requirements.txt

# Linux/Mac
cat requirements.txt
```

### Step 5: Clean Up (Opsional)

Edit `requirements.txt` dan hapus lines yang tidak perlu seperti:

- `pkg-resources==0.0.0`
- `setuptools==68.0.0`
- `wheel==0.41.0`

## Test Install dari Requirements

### Test Uninstall & Reinstall

```bash
# Uninstall core packages
pip uninstall -y ultralytics flask opencv-python numpy streamlit

# Install dari requirements
pip install -r requirements.txt

# Verify installation
python -c "import ultralytics, flask, cv2, numpy, streamlit; print('✅ Success!')"
```

## Commands untuk Verifikasi

### Check Python Environment

```bash
# Python version
python --version

# Pip version
pip --version

# Virtual environment (if using)
where python  # Windows
which python  # Linux/Mac
```

### Check Specific Packages

```bash
# Check ultralytics
pip show ultralytics

# Check flask
pip show flask

# Check opencv
pip show opencv-python

# Check streamlit
pip show streamlit

# Check torch
pip show torch
```

### Test Import All Dependencies

```python
# Save sebagai test_imports.py
try:
    import ultralytics
    import flask
    import cv2
    import numpy as np
    import requests
    import streamlit
    import torch
    import jupyter
    print("✅ All dependencies imported successfully!")

    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"Flask: {flask.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Streamlit: {streamlit.__version__}")

except ImportError as e:
    print(f"❌ Import error: {e}")
```

Jalankan dengan:

```bash
python test_imports.py
```

## Expected Output untuk Project Ini

Setelah menjalankan `pip list`, seharusnya ada packages seperti:

```
ultralytics           8.0.196
flask                 2.3.3
flask-cors            4.0.0
opencv-python         4.8.1.78
numpy                 1.24.3
requests              2.31.0
streamlit             1.28.0
torch                 2.0.1
torchvision           0.15.2
Pillow                10.0.0
jupyter               1.0.0
```

## Final Requirements File Example

```txt
# Machine Learning & Computer Vision
ultralytics==8.0.196
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.0

# Web Framework & API
flask==2.3.3
flask-cors==4.0.0
requests==2.31.0

# Frontend
streamlit==1.28.0

# Development & Utilities
jupyter==1.0.0
nbconvert==7.8.0
python-dotenv==1.0.0
```

## One-Liner Commands

### Generate Requirements (Windows)

```cmd
pip freeze | findstr /i "ultralytics flask opencv numpy streamlit torch requests jupyter pillow" > requirements.txt && type requirements.txt
```

### Generate Requirements (Linux/Mac)

```bash
pip freeze | grep -E "(ultralytics|flask|opencv|numpy|streamlit|torch|requests|jupyter|pillow)" > requirements.txt && cat requirements.txt
```

### Install and Test

```bash
pip install -r requirements.txt && python -c "import ultralytics, flask, cv2, numpy, streamlit; print('✅ All dependencies installed!')"
```

## Troubleshooting Quick Commands

### If pip freeze gives too many packages:

```bash
# List only manually installed packages
pip list --user --format=freeze > requirements.txt

# Or use pipreqs (install first)
pip install pipreqs
pipreqs . --force
```

### If there are version conflicts:

```bash
# Install with specific versions
pip install ultralytics==8.0.196 flask==2.3.3 opencv-python==4.8.1.78 numpy==1.24.3 streamlit==1.28.0

# Or use --upgrade flag
pip install -r requirements.txt --upgrade
```

### If packages not found:

```bash
# Update pip first
python -m pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt
```

## Next Steps After Requirements File

1. **Share requirements.txt** dengan team
2. **Create virtual environment** untuk production
3. **Deploy aplikasi** dengan `pip install -r requirements.txt`
4. **Version control** requirements.txt ke git

File requirements.txt yang sudah dibuat bisa digunakan untuk:

- Install dependencies di komputer lain
- Deploy ke server/cloud
- Share dengan team developers
- Setup CI/CD pipeline
