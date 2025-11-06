import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
# Try to import YOLO, but handle ImportError gracefully
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERROR = str(e)
    # Create a dummy YOLO class for type hints
    class YOLO:
        def __init__(self, *args, **kwargs):
            raise ImportError("YOLO not available")

# ==================== CONFIGURATION ====================
GROQ_API_KEY = "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"
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
HUGGINGFACE_API_URL = "https://huggingface.co/spaces/wanndev14/yolo-api"
MODEL_PATH = "runs/detect/train2/weights/best.pt"

INDONESIAN_FOOD_CLASSES = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

# ==================== STREAMLIT CONFIG ====================
st.set_page_config(
    page_title="Deteksi Gizi Makanan",
    page_icon="🍽️",
    layout="wide"
)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    """Load model dengan fallback handling"""
    try:
        if not YOLO_AVAILABLE:
            st.warning("⚠️ YOLO/OpenCV tidak tersedia di environment ini")
            st.info("🔄 Menggunakan API mode saja")
            return None
            
        if os.path.exists(MODEL_PATH):
            st.info("🔄 Loading model lokal...")
            model = YOLO(MODEL_PATH)
            st.success("✅ Model lokal berhasil diload")
            return model
        else:
            st.warning("⚠️ Model lokal tidak ditemukan, menggunakan API mode")
            return None
    except Exception as e:
        st.error(f"❌ Gagal load model lokal: {str(e)}")
        st.info("🔄 Fallback ke API mode")
        return None

# ==================== DETECTION FUNCTIONS ====================
def detect_with_api(image_data, api_url=HUGGINGFACE_API_URL):
    """Deteksi menggunakan HuggingFace API"""
    try:
        files = {"image": image_data}
        response = requests.post(f"{api_url}/detect-gizi", files=files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ API Connection Error: {str(e)}")
        return None

def detect_food_local(model, image, confidence_threshold=0.5):
    """Deteksi menggunakan model lokal"""
    try:
        # Convert image to RGB array for YOLO
        image_array = np.array(image.convert('RGB'))
        
        # Run detection
        results = model(image_array)[0]
        
        detected_objects = []
        food_names = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            if confidence >= confidence_threshold and class_id < len(INDONESIAN_FOOD_CLASSES):
                class_name = INDONESIAN_FOOD_CLASSES[class_id]
                
                detected_objects.append({
                    "nama": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
                food_names.append(class_name)
        
        # Draw bounding boxes using PIL
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["bbox"]
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            
            # Draw label
            label = f"{obj['nama']} ({obj['confidence']:.2f})"
            if font:
                draw.text((x1, y1-20), label, fill=(0, 255, 0), font=font)
            else:
                draw.text((x1, y1-20), label, fill=(0, 255, 0))
        
        return detected_objects, food_names, annotated_image
        
    except Exception as e:
        st.error(f"❌ Local detection error: {str(e)}")
        return [], [], None

def get_nutritional_analysis(detected_foods):
    """Analisis gizi menggunakan Groq API"""
    if not GROQ_API_KEY:
        return "⚠️ Groq API key not configured."
    
    makanan_str = ', '.join(list(set(detected_foods)))
    if not makanan_str:
        return "Tidak ada makanan terdeteksi."
    
    prompt = f"Berikan informasi kandungan gizi dari: {makanan_str}. Jawab dalam tabel Bahasa Indonesia."
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "Kamu adalah ahli gizi makanan Indonesia."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
            },
            timeout=30,
        )
        response.raise_for_status()
        completion = response.json()
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error analisis gizi: {str(e)}"

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Load model sekali di awal
    model = load_model()
    
    # Status
    st.subheader("📊 Status")
    if not YOLO_AVAILABLE:
        st.error("❌ YOLO/OpenCV Tidak Tersedia")
        st.warning("🌐 Mode: API Only (HuggingFace)")
        st.info(f"🔗 API: {HUGGINGFACE_API_URL}")
        st.markdown("**Environment Streamlit Cloud tidak mendukung OpenCV**")
    elif model:
        st.success("✅ Mode: Local (Model Loaded)")
        if os.path.exists(MODEL_PATH):
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.info(f"📁 Model Size: {size_mb:.1f} MB")
    else:
        st.warning("🌐 Mode: API (HuggingFace)")
        st.info(f"🔗 API: {HUGGINGFACE_API_URL}")
    
    # Confidence threshold
    st.subheader("🔍 Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05
    )
    
    st.success("✅ Groq API: Ready")

# ==================== MAIN APP ====================
st.title("🍽️ Deteksi Gizi Makanan")
st.write("Upload foto makanan untuk deteksi otomatis dan analisis gizi")

# Tampilkan mode yang aktif
if not YOLO_AVAILABLE:
    st.warning("🌐 **API Only Mode** - YOLO/OpenCV tidak tersedia di environment ini")
    st.info("Menggunakan HuggingFace API untuk deteksi makanan")
elif model:
    st.success("🖥️ **Local Mode** - Menggunakan model YOLO lokal (Lebih Cepat)")
else:
    st.info("🌐 **API Mode** - Menggunakan HuggingFace API (Stabil)")

# File upload
uploaded_file = st.file_uploader("Pilih gambar makanan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Uploaded", use_container_width=True)
    
    if st.button("🔍 Deteksi & Analisis Gizi", type="primary", use_container_width=True):
        with st.spinner("🔄 Memproses..."):
            try:
                # Convert image
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                detected_objects = []
                food_names = []
                annotated_image = None
                
                # Pilih metode deteksi
                if model:
                    # Local detection
                    st.info("🖥️ Menggunakan model lokal...")
                    detected_objects, food_names, annotated_image = detect_food_local(
                        model, image, confidence_threshold
                    )
                else:
                    # API detection
                    st.info("🌐 Mengirim ke HuggingFace API...")
                    result = detect_with_api(img_bytes)
                    if result:
                        detected_objects = result.get("objects", [])
                        # Filter by confidence
                        detected_objects = [obj for obj in detected_objects if obj.get("confidence", 0) >= confidence_threshold]
                        food_names = [obj["nama"] for obj in detected_objects]
                        
                        # Handle annotated image from API
                        img_data_url = result.get("image", "")
                        if img_data_url:
                            try:
                                # Parse base64 data URL
                                header, encoded = img_data_url.split(",", 1)
                                img_data = base64.b64decode(encoded)
                                annotated_image = Image.open(io.BytesIO(img_data))
                            except Exception as img_error:
                                st.warning(f"⚠️ Could not load annotated image: {img_error}")
                                pass
                
                # Get nutrition analysis
                with st.spinner("🥗 Menganalisis gizi..."):
                    nutritional_info = get_nutritional_analysis(food_names)
                
                # ==================== DISPLAY RESULTS ====================
                st.subheader("🎯 Hasil Deteksi")
                
                # Statistics
                if detected_objects:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Makanan", len(detected_objects))
                    with col2:
                        st.metric("Jenis Makanan", len(set(food_names)))
                    with col3:
                        avg_conf = sum(obj['confidence'] for obj in detected_objects) / len(detected_objects)
                        st.metric("Rata Confidence", f"{avg_conf:.3f}")
                    
                    # Annotated Image
                    st.subheader("📷 Hasil Visual")
                    if annotated_image is not None:
                        st.image(annotated_image, use_container_width=True)
                    
                    # Detection Details
                    st.subheader("📦 Detail Makanan")
                    detection_data = []
                    for i, obj in enumerate(detected_objects, 1):
                        detection_data.append({
                            "Makanan": obj["nama"],
                            "Confidence": f"{obj['confidence']:.3f}",
                            "Posisi": f"({obj['bbox'][0]}, {obj['bbox'][1]})"
                        })
                    
                    df = pd.DataFrame(detection_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Food tags
                    st.subheader("🏷️ Makanan Terdeteksi")
                    for food in set(food_names):
                        st.markdown(f"`{food}`", unsafe_allow_html=True)
                    
                    # Nutrition Analysis
                    st.subheader("🥗 Analisis Gizi")
                    if nutritional_info:
                        st.markdown(nutritional_info)
                    else:
                        st.info("Tidak ada analisis gizi")
                        
                    st.success("✅ Analisis selesai!")
                    
                else:
                    st.warning("❌ Tidak ada makanan terdeteksi")
                    if nutritional_info and "tidak ada" not in nutritional_info.lower():
                        st.subheader("🥗 Analisis Gizi")
                        st.markdown(nutritional_info)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Deteksi Gizi Makanan** | YOLO + Groq API")