# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

Indonesian food detection system using YOLOv11 + Groq LLM for nutritional analysis. Streamlit frontend + Flask API backend.

## Critical Commands

### Running the Application

```bash
# Start Flask API server (required for Streamlit to work)
python api.ipynb  # or run in Jupyter
# Runs on http://localhost:5000

# Start Streamlit frontend
streamlit run app.py
# Runs on http://localhost:8501
```

### Model Training & Validation

```bash
# Train YOLO model (run in Jupyter)
python train.ipynb

# Validate model performance
model.val(data="dataset/data.yaml", split="val", imgsz=224, batch=16, conf=0.001, iou=0.65, plots=True)
```

## Non-Obvious Architecture Patterns

### Model Loading

- **Critical**: YOLO model path is hardcoded to `runs/detect/train2/weights/best.pt` in [`api.ipynb`](api.ipynb:33)
- API will fail if this file doesn't exist or path changes
- Class names are duplicated between [`api.ipynb`](api.ipynb:39-49) and [`dataset/data.yaml`](dataset/data.yaml:6) - must stay synchronized

### API Integration

- Groq API key is hardcoded in [`api.ipynb`](api.ipynb:27) - should be environment variable in production
- Model uses `meta-llama/llama-4-maverick-17b-128e-instruct` for nutritional analysis
- API timeout is 90 seconds for image processing + LLM inference

### Data Processing

- Images are preprocessed with OpenCV, not PIL (despite Streamlit using PIL for display)
- YOLO format: `class_id center_x center_y width height` (normalized coordinates 0-1)
- Dataset uses Roboflow export with 29 Indonesian food classes

### Streamlit Configuration

- Confidence filter in UI only affects display, NOT actual detection (filtering happens client-side)
- API base URL defaults to `http://localhost:5000` but must be configured in sidebar
- Image upload accepts JPG/PNG but API processes all as JPEG internally

## Dataset Structure

- Train/val/test split defined in [`dataset/data.yaml`](dataset/data.yaml:1-3)
- Images resized to 640x640 during preprocessing (Roboflow export)
- Labels use YOLOv11 format with normalized coordinates

## Gotchas

- API must be running before Streamlit app can detect food
- Model validation requires specific parameters: `conf=0.001, iou=0.65` for accurate results
- Image annotations returned as base64 data URLs, not direct file paths
- Nutritional analysis is in Indonesian language only
