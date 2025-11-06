# -*- coding: utf-8 -*-
"""
Streamlit-Compatible Food Detection API
Bisa di-deploy langsung ke Streamlit Cloud tanpa Flask server
"""

import streamlit as st
import numpy as np
import os
import cv2
import requests
import io
import base64
from ultralytics import YOLO
from PIL import Image
import tempfile

# Configuration
GROQ_API_KEY = "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"
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Load model (cached for performance)
@st.cache_resource
def load_model():
    model_path = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")
    return YOLO(model_path)

# Class names
CLASS_NAMES = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

def preprocess_image(image_data):
    """Preprocess image for YOLO detection"""
    if isinstance(image_data, np.ndarray):
        return image_data
    else:
        # Convert PIL Image to numpy array
        image = np.array(image_data)
        return image

def draw_boxes(image, results):
    """Draw bounding boxes on image"""
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = CLASS_NAMES[cls]
        conf = float(box.conf[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return image

def get_nutritional_analysis(detected_foods):
    """Get nutritional analysis from Groq API"""
    makanan_str = ', '.join(list(set(detected_foods)))
    prompt = f"Berikan informasi kandungan gizi dari makanan berikut: {makanan_str}. Jawab dalam bentuk tabel dan bahasa Indonesia."
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "Kamu adalah asisten gizi makanan Indonesia."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
            },
            timeout=30,
        )
        completion = response.json()
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error getting nutritional analysis: {str(e)}"

def detect_food(image):
    """Main food detection function"""
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Run YOLO detection
    model = load_model()
    results = model(processed_image)[0]
    
    # Extract detected objects
    detected_objects = []
    makanan_list = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = CLASS_NAMES[cls]
        conf = float(box.conf[0])
        
        makanan_list.append(label)
        detected_objects.append({
            "nama": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
    
    # Get nutritional analysis
    gizi_text = get_nutritional_analysis(makanan_list)
    
    # Draw boxes on image
    boxed_image = draw_boxes(processed_image.copy(), results)
    
    # Convert to base64 for display
    _, img_encoded = cv2.imencode('.jpg', boxed_image)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
    
    return {
        "objects": detected_objects,
        "image": "data:image/jpeg;base64," + img_base64,
        "gizi": gizi_text,
        "annotated_image": boxed_image
    }

# Streamlit App
def main():
    st.set_page_config(
        page_title="Food Detection & Nutrition Analysis",
        page_icon="🍽️",
        layout="wide"
    )
    
    st.title("🍽️ Deteksi Makanan Indonesia & Analisis Gizi")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🔧 Konfigurasi")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Gambar")
        uploaded_file = st.file_uploader(
            "Pilih gambar makanan:", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar makanan Indonesia untuk dideteksi"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Original", use_column_width=True)
            
            # Detect button
            if st.button("🔍 Deteksi Makanan", type="primary"):
                with st.spinner("Sedang mendeteksi makanan..."):
                    try:
                        # Convert PIL to numpy array
                        image_array = np.array(image)
                        
                        # Run detection
                        results = detect_food(image_array)
                        
                        # Store results in session state
                        st.session_state.detection_results = results
                        
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
    
    with col2:
        st.subheader("📊 Hasil Deteksi")
        
        if 'detection_results' in st.session_state:
            results = st.session_state.detection_results
            
            # Display annotated image
            st.image(results["annotated_image"], caption="Hasil Deteksi", use_column_width=True)
            
            # Display detected objects
            st.subheader("🍱 Makanan Terdeteksi")
            for obj in results["objects"]:
                if obj["confidence"] >= confidence_threshold:
                    st.success(f"**{obj['nama']}** - Confidence: {obj['confidence']:.2f}")
            
            # Display nutritional analysis
            st.subheader("🥗 Analisis Gizi")
            st.markdown(results["gizi"])
            
            # Export button
            if st.button("📥 Export Results"):
                export_data = {
                    "detected_objects": results["objects"],
                    "nutritional_analysis": results["gizi"]
                }
                st.download_button(
                    label="Download JSON",
                    data=str(export_data),
                    file_name="food_detection_results.json",
                    mime="application/json"
                )
        else:
            st.info("Silakan upload gambar terlebih dahulu untuk memulai deteksi.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Model:** YOLOv11 Food Detection | **LLM:** Groq Llama-4-Maverick | **Classes:** 29 Makanan Indonesia")

# API Endpoint untuk external calls
def api_detect_gizi(image_file):
    """API endpoint yang bisa dipanggil dari luar"""
    try:
        # Read image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Run detection
        results = detect_food(image_array)
        
        return {
            "success": True,
            "data": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    main()