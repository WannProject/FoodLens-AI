# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Debug Rules (Non-Obvious Only)

### API Startup Issues

- Flask API will fail silently if YOLO model file doesn't exist at `runs/detect/train2/weights/best.pt`
- Check model file existence before starting API server
- API runs on port 5000 - ensure no conflicting services

### Common Failure Points

- **Groq API**: 30-second timeout for LLM calls - check network connectivity
- **Image Processing**: OpenCV vs PIL format mismatches cause silent failures
- **Base64 Encoding**: Image data URLs must follow exact format `data:image/jpeg;base64,<data>`
- **Class Name Mismatch**: API class names vs dataset config causes detection failures

### Debugging Model Detection

- Use `conf=0.001, iou=0.65` for validation - different values give poor results
- Model expects 224x224 input regardless of training image size
- Check label format: normalized coordinates (0-1 range) not pixel values

### API Response Debugging

- Expected JSON structure: `{"objects": [...], "image": "data:...", "gizi": "..."}`
- Empty objects array usually means model confidence threshold too high
- Missing gizi field indicates Groq API failure

### Streamlit Debugging

- API connection errors display generic message - check actual API status
- Confidence filter affects display only, not underlying detection
- Image upload accepts PNG but API processes as JPEG - check format conversion

### Environment Dependencies

- Requires `ultralytics`, `flask`, `flask_cors`, `streamlit`, `opencv-python`, `requests`
- Groq API key hardcoded in [`api.ipynb`](api.ipynb:27) - check validity
- Model validation requires specific YOLOv11 version compatibility

### Silent Failure Patterns

- Missing model file causes API to start but fail on requests
- Incorrect image format returns 200 OK with empty results
- Network timeouts to Groq API hang entire request pipeline
