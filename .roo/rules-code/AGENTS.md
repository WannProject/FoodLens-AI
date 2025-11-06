# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Coding Rules (Non-Obvious Only)

### Model Path Management

- **CRITICAL**: YOLO model path is hardcoded in [`api.ipynb`](api.ipynb:33) - always update this if model location changes
- Model file must exist at `runs/detect/train2/weights/best.pt` or API will fail on startup
- Class names array in [`api.ipynb`](api.ipynb:39-49) MUST match [`dataset/data.yaml`](dataset/data.yaml:6) exactly

### API Key Security

- Groq API key is hardcoded in [`api.ipynb`](api.ipynb:27) - replace with environment variable in production
- API key is used for `meta-llama/llama-4-maverick-17b-128e-instruct` model

### Image Processing

- Use OpenCV for preprocessing in API (`cv2.imdecode`), NOT PIL
- Streamlit frontend uses PIL for display - handle format conversion properly
- All images processed as JPEG internally regardless of input format

### Data Format Conventions

- YOLO label format: `class_id center_x center_y width height` (normalized 0-1)
- Bounding boxes returned as `[x1, y1, x2, y2]` pixel coordinates
- Image annotations returned as base64 data URLs, not file paths

### Streamlit API Integration

- API timeout is 90 seconds - handle timeouts gracefully in frontend
- Confidence filtering happens client-side, doesn't affect actual detection results
- Default API URL is `http://localhost:5000` but configurable in sidebar

### Error Handling Patterns

- API returns JSON with `objects`, `image`, and `gizi` fields
- Streamlit must handle ConnectionError, Timeout, and HTTPError separately
- Base64 image parsing requires specific format handling

### Model Training Parameters

- Validation requires specific params: `conf=0.001, iou=0.65` for accurate results
- Training uses 224x224 images regardless of dataset preprocessing
- Model trained for 25 epochs with batch size 16
