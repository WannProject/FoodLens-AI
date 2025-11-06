# Quick Install Commands

## One-Liner Install (Copy & Paste)

### Windows (PowerShell/CMD)

```powershell
# Install semua dependencies dalam satu command
pip install ultralytics==8.0.196 flask==2.3.3 flask-cors==4.0.0 opencv-python==4.8.1.78 numpy==1.24.3 requests==2.31.0 Pillow==10.0.0 streamlit==1.28.0 jupyter==1.0.0 nbconvert==7.8.0 python-dotenv==1.0.0 torch==2.0.1 torchvision==0.15.2
```

### Linux/Mac

```bash
# Install semua dependencies dalam satu command
pip install ultralytics==8.0.196 flask==2.3.3 flask-cors==4.0.0 opencv-python==4.8.1.78 numpy==1.24.3 requests==2.31.0 Pillow==10.0.0 streamlit==1.28.0 jupyter==1.0.0 nbconvert==7.8.0 python-dotenv==1.0.0 torch==2.0.1 torchvision==0.15.2
```

## Step by Step Install

### Step 1: Virtual Environment (Recommended)

```bash
# Windows
python -m venv food_detection_env
food_detection_env\Scripts\activate

# Linux/Mac
python3 -m venv food_detection_env
source food_detection_env/bin/activate
```

### Step 2: Install Core ML Dependencies

```bash
pip install ultralytics==8.0.196 torch==2.0.1 torchvision==0.15.2
```

### Step 3: Install Web Framework

```bash
pip install flask==2.3.3 flask-cors==4.0.0 requests==2.31.0
```

### Step 4: Install Image Processing

```bash
pip install opencv-python==4.8.1.78 numpy==1.24.3 Pillow==10.0.0
```

### Step 5: Install Frontend

```bash
pip install streamlit==1.28.0
```

### Step 6: Install Development Tools

```bash
pip install jupyter==1.0.0 nbconvert==7.8.0 python-dotenv==1.0.0
```

## Install dengan Upgrade

```bash
# Upgrade pip dulu
python -m pip install --upgrade pip

# Install dengan upgrade semua package
pip install --upgrade ultralytics flask flask-cors opencv-python numpy requests Pillow streamlit jupyter nbconvert python-dotenv torch torchvision
```

## Verify Installation

```bash
# Test semua imports
python -c "
import ultralytics
import flask
import cv2
import numpy as np
import requests
import streamlit
import torch
import jupyter
print('✅ All dependencies installed successfully!')
print(f'Ultralytics: {ultralytics.__version__}')
print(f'Flask: {flask.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'Streamlit: {streamlit.__version__}')
"
```

## Alternative: Install with Latest Versions

```bash
# Install versi terbaru tanpa lock version
pip install ultralytics flask flask-cors opencv-python numpy requests Pillow streamlit jupyter nbconvert python-dotenv torch torchvision
```

## Common Installation Issues & Solutions

### Issue 1: PyTorch CUDA Error

```bash
# Solution: Install CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue 2: OpenCV Import Error

```bash
# Solution: Install headless version
pip uninstall opencv-python
pip install opencv-python-headless
```

### Issue 3: Ultralytics Download Error

```bash
# Solution: Install from specific source
pip install ultralytics --extra-index-url https://download.pytorch.org/whl/cpu
```

### Issue 4: Streamlit CORS Issues

```bash
# Solution: Install streamlit dengan semua dependencies
pip install streamlit[all]
```

## Minimum Requirements untuk Testing

```bash
# Install minimal dependencies untuk testing API saja
pip install ultralytics flask flask-cors opencv-python numpy requests

# Install minimal dependencies untuk testing Frontend saja
pip install streamlit Pillow
```

## Check Current Installed Packages

```bash
# List all installed packages
pip list

# Check specific package versions
pip show ultralytics flask streamlit opencv-python
```

## Installation Time Estimates

- **Fast Install** (with good internet): 2-5 minutes
- **Normal Install**: 5-10 minutes
- **Slow Install** (poor internet): 10-20 minutes
- **PyTorch install**: Usually the longest part (1-5 minutes)

## After Installation

### 1. Konversi api.ipynb ke api.py

```bash
# Method 1: Manual copy paste (recommended)
# Copy kode dari solusi-error-api.md

# Method 2: Using nbconvert
pip install nbconvert
jupyter nbconvert --to python api.ipynb
```

### 2. Test API Server

```bash
python api.py
# Should run on http://localhost:5000
```

### 3. Test Frontend

```bash
streamlit run app.py
# Should run on http://localhost:8501
```

### 4. Test Full Pipeline

1. Buka http://localhost:8501
2. Upload gambar makanan
3. Check detection results
