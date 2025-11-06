# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Architecture Rules (Non-Obvious Only)

### System Architecture Constraints

- **Tight Coupling**: Streamlit frontend cannot work without Flask API running first
- **Stateless Design**: YOLO model loaded once on API startup, shared across requests
- **Sequential Processing**: Detection → LLM analysis → Image annotation must complete in order
- **Single Point of Failure**: Groq API dependency blocks entire nutritional analysis pipeline

### Data Flow Architecture

- **Image Processing Pipeline**: Upload → OpenCV decode → YOLO detection → OpenCV annotation → Base64 encode
- **Mixed Libraries**: OpenCV for processing (API) vs PIL for display (Streamlit) - intentional separation
- **Synchronous Processing**: No async patterns - LLM calls block entire request thread
- **Memory Constraints**: Full image processing happens in RAM, no disk caching

### Model Integration Patterns

- **Hardcoded Dependencies**: Model path and class names duplicated between files
- **Version Locking**: Specific YOLOv11 weights format required - cannot upgrade independently
- **Training/Inference Mismatch**: Training uses 640x640, inference uses 224x224
- **No Model Versioning**: Single `best.pt` file represents production model

### API Design Constraints

- **Monolithic Endpoint**: `/detect-gizi` handles detection, LLM, and annotation together
- **Timeout Dependencies**: 90-second timeout must accommodate both YOLO and LLM processing
- **Error Propagation**: LLM failures don't prevent detection results from returning
- **No Retry Logic**: Single attempt at Groq API call, no fallback mechanisms

### Frontend Architecture Limitations

- **Configuration in UI**: API URL and settings only configurable in Streamlit sidebar
- **Client-side Filtering**: Confidence filtering doesn't reduce server processing load
- **No State Management**: Each upload is independent, no session persistence
- **Display Only**: Streamlit app cannot trigger model retraining or updates

### Deployment Architecture

- **Two-Service Requirement**: Must deploy both Flask API and Streamlit frontend
- **Port Dependencies**: API hardcoded to port 5000, frontend expects this
- **No Containerization**: No Docker files or deployment configs provided
- **Local File Dependencies**: Model weights must exist in specific filesystem paths

### Scalability Constraints

- **Single Model Instance**: No model batching or concurrent processing
- **Groq Rate Limits**: LLM API calls subject to external rate limiting
- **Memory Bound**: Full image processing pipeline loads entire images in memory
- **No Load Balancing**: Single Flask instance handles all requests
