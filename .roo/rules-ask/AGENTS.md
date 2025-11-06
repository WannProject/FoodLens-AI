# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Documentation Rules (Non-Obvious Only)

### Counterintuitive Architecture

- **Jupyter Notebooks as Services**: [`api.ipynb`](api.ipynb) and [`train.ipynb`](train.ipynb) are meant to be run as services, not just notebooks
- **Mixed Image Processing**: API uses OpenCV while Streamlit frontend uses PIL - two different image libraries
- **Hardcoded Production Values**: API key and model paths are hardcoded despite being production-ready code

### Misleading File Organization

- `dataset/` contains preprocessed Roboflow export, not raw training data
- `runs/` contains training artifacts, not runtime logs
- Model weights in `runs/detect/train2/weights/best.pt` are the active production model
- `yolo11n.pt` in root is the base pretrained model, not the trained model

### Hidden Dependencies

- Groq API integration is undocumented but essential for nutritional analysis
- YOLOv11 specific version required for compatibility with trained weights
- Indonesian language hardcoded in LLM prompts - not configurable

### Undocumented Workflows

- Must start Flask API server before Streamlit app can function
- Model training requires Jupyter environment, not standard Python scripts
- Validation has very specific parameters (`conf=0.001, iou=0.65`) that aren't obvious

### API Behavior Contradictions

- Confidence filter in UI doesn't affect actual detection results
- Image upload accepts multiple formats but always processes as JPEG
- API returns base64 encoded images, not file paths
- Nutritional analysis always in Indonesian regardless of user preference

### Dataset Specifics

- 29 Indonesian food classes defined in both [`api.ipynb`](api.ipynb:39-49) and [`dataset/data.yaml`](dataset/data.yaml:6)
- Images pre-resized to 640x640 by Roboflow, not during training
- Labels use normalized coordinates (0-1) despite pixel-based output from detection

### Configuration Gotchas

- Training uses 224x224 input size, different from dataset preprocessing size
- Model trained for 25 epochs with specific augmentation parameters
- No requirements.txt file - dependencies must be inferred from imports
