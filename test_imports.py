#!/usr/bin/env python3
"""
Test script to verify import handling for Streamlit Cloud deployment
"""

print("Testing imports...")

# Test YOLO import with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO import successful")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"❌ YOLO import failed: {e}")

# Test OpenCV import
try:
    import cv2
    print(f"✅ OpenCV import successful (version: {cv2.__version__})")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

# Test model file existence
import os
MODEL_PATH = "runs/detect/train2/weights/best.pt"
if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model file found: {MODEL_PATH} ({size_mb:.1f} MB)")
else:
    print(f"❌ Model file not found: {MODEL_PATH}")

# Test Streamlit import
try:
    import streamlit as st
    print(f"✅ Streamlit import successful (version: {st.__version__})")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

print("\n=== Summary ===")
if YOLO_AVAILABLE and os.path.exists(MODEL_PATH):
    print("🟢 Local mode should work (YOLO + model available)")
elif not YOLO_AVAILABLE:
    print("🟡 API-only mode (YOLO not available, but app should still work)")
else:
    print("🔴 Issues detected - check above errors")