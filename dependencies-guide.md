# Dependencies Guide - Food Detection App

## Daftar Dependencies Lengkap

### Core Dependencies (API Backend)

```txt
# Machine Learning & Computer Vision
ultralytics>=8.0.196          # YOLOv11 framework
opencv-python>=4.8.1.78      # Image processing
numpy>=1.24.3                # Numerical operations
torch>=2.0.1                 # PyTorch backend untuk YOLO
torchvision>=0.15.2          # Computer vision utilities

# Web Framework
flask>=2.3.3                 # Web API framework
flask-cors>=4.0.0            # CORS support
requests>=2.31.0             # HTTP client untuk Groq API

# Image Processing & Utils
Pillow>=10.0.0               # Image processing (backend Streamlit)
```

### Frontend Dependencies (Streamlit)

```txt
# Streamlit Framework
streamlit>=1.28.0            # Frontend framework
streamlit-extras>=0.3.5      # Additional Streamlit components
plotly>=5.15.0               # Interactive charts (opsional)
```

### Development & Utilities

```txt
# Development Tools
jupyter>=1.0.0               # Jupyter notebook support
nbconvert>=7.8.0             # Notebook conversion
matplotlib>=3.7.2            # Plotting dan visualization
seaborn>=0.12.2              # Statistical visualization

# Environment Management
python-dotenv>=1.0.0         # Environment variables management
```

## Cara Install Dependencies

### Opsi 1: Install Semua Dependencies

```bash
# Buat virtual environment (recommended)
python -m venv food_detection_env

# Aktifkan virtual environment
# Windows:
food_detection_env\Scripts\activate
# Linux/Mac:
source food_detection_env/bin/activate

# Install dependencies
pip install ultralytics>=8.0.196 flask>=2.3.3 flask-cors>=4.0.0 opencv-python>=4.8.1.78 numpy>=1.24.3 requests>=2.31.0 Pillow>=10.0.0 streamlit>=1.28.0 jupyter>=1.0.0 nbconvert>=7.8.0 python-dotenv>=1.0.0 torch>=2.0.1 torchvision>=0.15.2
```

### Opsi 2: Install dengan Requirements File

```bash
# Buat file requirements.txt terlebih dahulu (lihat konten di bawah)
pip install -r requirements.txt
```

### Opsi 3: Install Gradual

```bash
# Install ML dependencies dulu
pip install ultralytics torch torchvision

# Install web framework
pip install flask flask-cors requests

# Install image processing
pip install opencv-python numpy Pillow

# Install frontend
pip install streamlit

# Install development tools
pip install jupyter nbconvert python-dotenv
```

## File requirements.txt (Copy Paste)

```txt
# Machine Learning & Computer Vision
ultralytics==8.0.196
opencv-python==4.8.1.78
numpy==1.24.3
torch==2.0.1
torchvision==0.15.2

# Web Framework
flask==2.3.3
flask-cors==4.0.0
requests==2.31.0

# Image Processing
Pillow==10.0.0

# Frontend
streamlit==1.28.0

# Development Tools
jupyter==1.0.0
nbconvert==7.8.0
python-dotenv==1.0.0
```

## Check Dependencies Status

### Cek Versi yang Terinstall

```bash
# Cek semua installed packages
pip list

# Cek spesifik package
pip show ultralytics
pip show flask
pip show streamlit
```

### Cek Kompatibilitas

```python
# Test import dependencies
import ultralytics
import flask
import cv2
import numpy as np
import requests
import streamlit
import torch

print("✅ All dependencies imported successfully!")
print(f"Ultralytics version: {ultralytics.__version__}")
print(f"Flask version: {flask.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
```

## Troubleshooting Dependencies

### Common Issues & Solutions

1. **PyTorch CUDA Issues**

```bash
# Install CPU version jika GPU tidak tersedia
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. **OpenCV Headless Issues**

```bash
# Install headless version untuk server
pip install opencv-python-headless
```

3. **Ultralytics Import Error**

```bash
# Reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics==8.0.196
```

4. **Streamlit CORS Issues**

```bash
# Install streamlit dengan dependencies lengkap
pip install streamlit[all]
```

## Version Compatibility Matrix

| Component   | Minimum Version | Recommended Version | Notes                                        |
| ----------- | --------------- | ------------------- | -------------------------------------------- |
| Python      | 3.8             | 3.9-3.11            | Python 3.12+ might have compatibility issues |
| Ultralytics | 8.0.0           | 8.0.196             | YOLOv11 support                              |
| PyTorch     | 1.12            | 2.0.1+              | CUDA support optional                        |
| OpenCV      | 4.5.0           | 4.8.1+              | CPU version sufficient                       |
| Flask       | 2.0             | 2.3.3+              | CORS support important                       |
| Streamlit   | 1.25            | 1.28.0+             | Latest stable version                        |

## Virtual Environment Setup (Recommended)

### Windows

```cmd
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Linux/Mac

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

## Production Considerations

### Docker Dependencies

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "api.py"]
```

### Environment Variables

```bash
# .env file
GROQ_API_KEY=your_api_key_here
MODEL_PATH=runs/detect/train2/weights/best.pt
FLASK_ENV=development
FLASK_DEBUG=True
```
