import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from typing import Tuple, List, Dict, Optional
import numpy as np
import os
import tempfile
import time
from datetime import datetime

# Try to import OpenCV and YOLO, but handle ImportError gracefully
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    YOLO_AVAILABLE = False
    # Create a dummy YOLO class for type hints
    class YOLO:
        def __init__(self, *args, **kwargs):
            raise ImportError("YOLO not available")

# Optional Ngrok import
try:
    import pyngrok
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    pyngrok = None
    ngrok = None
from datetime import datetime

# -----------------------------
# Configuration & Constants
# -----------------------------
# Hardcoded API Key - FIX untuk error st.secrets
GROQ_API_KEY = "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"
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Indonesian food classes (29 classes)
INDONESIAN_FOOD_CLASSES = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

# Default model path (absolute path to avoid relative path issues)
DEFAULT_MODEL_PATH = os.path.join(os.getcwd(), "runs", "detect", "train2", "weights", "best.pt")

# Colors for bounding boxes (BGR format for OpenCV)
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (0, 0, 0)   # Black

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="DataLens-AI Deteksi Makanan",
    page_icon="🍽️",
    layout="wide",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "UI Streamlit untuk Deteksi Gizi Makanan (YOLO + LLM)"
    }
)

# -----------------------------
# Session State Management
# -----------------------------
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'upload_initialized': False,
        'current_file': None,
        'processing_error': None,
        'model_loaded': False,
        'model_path': DEFAULT_MODEL_PATH,
        'detection_results': None,
        'current_image': None,
        'ngrok_url': None,
        'ngrok_active': False,
        'processing': False,
        'error_message': None,
        'success_message': None,
        'confidence_threshold': 0.5,
        'last_detection_time': None,
        'detection_history': [],
        'mode': 'Local Mode'
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize session state early
try:
    initialize_session_state()
    st.session_state.upload_initialized = True
    
    # Auto-load model if YOLO is available and model not loaded yet
    if YOLO_AVAILABLE and not st.session_state.model_loaded:
        st.write("🔄 **Loading YOLO model...**")
        model = load_yolo_model(st.session_state.model_path)
        if model:
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("✅ **Model loaded successfully!**")
            st.rerun()
        else:
            st.error("❌ **Failed to load model**")
                
except Exception as e:
    st.error(f"Session initialization error: {str(e)}")
    st.session_state.upload_initialized = False

# -----------------------------
# Custom CSS Styles
# -----------------------------
def inject_custom_styles():
    """Inject custom CSS for better UI"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .chip {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px 6px 0 0;
        border-radius: 16px;
        background: #e3f2fd;
        color: #1976d2;
        font-size: 12px;
        border: 1px solid #bbdefb;
        white-space: nowrap;
    }
    
    .metric-card {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #4caf50; }
    .status-offline { background-color: #f44336; }
    .status-loading { background-color: #ff9800; }
    
    .ngrok-url {
        background: #f0f8ff;
        border: 1px solid #4169e1;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        font-family: monospace;
        word-break: break-all;
    }
    
    .download-btn {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 25px;
        cursor: pointer;
        font-weight: bold;
        text-decoration: none;
        display: inline-block;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Model Loading Functions
# -----------------------------
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo_model(model_path: str) -> Optional[YOLO]:
    """
    Load YOLO model with caching for performance
    """
    try:
        st.write(f"🔍 **Checking YOLO availability...**")
        if not YOLO_AVAILABLE:
            st.error("❌ YOLO/OpenCV tidak tersedia di environment ini")
            st.info("🔄 Menggunakan API mode atau demo mode saja")
            return None
            
        st.write(f"📁 **Model path:** {model_path}")
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None
            
        st.write(f"📦 **Loading YOLO model...**")
        model = YOLO(model_path)
        st.success(f"✅ **Model loaded successfully!**")
        st.info(f"📍 **Model location:** {model_path}")
        return model
        
    except Exception as e:
        st.error(f"❌ **Error loading model:** {str(e)}")
        st.write(f"🐛 **Debug info:**")
        st.write(f"- Model path: {model_path}")
        st.write(f"- Path exists: {os.path.exists(model_path)}")
        st.write(f"- YOLO available: {YOLO_AVAILABLE}")
        return None

# -----------------------------
# Image Processing Functions
# -----------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess PIL image for YOLO detection
    """
    # Convert PIL to numpy array
    image_array = np.array(image.convert('RGB'))
    
    # Only convert to BGR if OpenCV is available
    if CV2_AVAILABLE and image_array.shape[2] == 3:  # RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_array

def detect_food_objects(model: YOLO, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Perform food detection using YOLO model
    """
    try:
        # Run inference
        results = model(image)[0]
        
        detected_objects = []
        food_names = []
        
        for box in results.boxes:
            # Extract bounding box and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Filter by confidence threshold
            if confidence >= confidence_threshold:
                # Get class name
                if class_id < len(INDONESIAN_FOOD_CLASSES):
                    class_name = INDONESIAN_FOOD_CLASSES[class_id]
                else:
                    class_name = f"Unknown_{class_id}"
                
                detected_objects.append({
                    "nama": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id
                })
                food_names.append(class_name)
        
        return detected_objects, food_names
        
    except Exception as e:
        st.error(f"❌ Error during detection: {str(e)}")
        return [], []

def draw_bounding_boxes_opencv(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on image using OpenCV
    """
    if not CV2_AVAILABLE:
        return draw_bounding_boxes_pil(image, detections)
        
    # Create a copy to avoid modifying the original
    annotated_image = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]
        class_name = detection["nama"]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        
        # Prepare label text
        label = f"{class_name} ({confidence:.2f})"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(
            annotated_image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            BOX_COLOR,
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            TEXT_COLOR,
            thickness
        )
    
    return annotated_image

def draw_bounding_boxes_pil(image_array: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on image using PIL (fallback when OpenCV not available)
    """
    # Convert numpy array back to PIL Image
    if CV2_AVAILABLE and len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert BGR to RGB if it's BGR format
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_array
        
    pil_image = Image.fromarray(image_rgb.astype(np.uint8))
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]
        class_name = detection["nama"]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        
        # Draw label
        label = f"{class_name} ({confidence:.2f})"
        if font:
            draw.text((x1, y1-20), label, fill=(0, 255, 0), font=font)
        else:
            draw.text((x1, y1-20), label, fill=(0, 255, 0))
    
    # Convert back to numpy array
    return np.array(pil_image)

def draw_bounding_boxes(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on image (with automatic fallback)
    """
    if CV2_AVAILABLE:
        return draw_bounding_boxes_opencv(image, detections)
    else:
        return draw_bounding_boxes_pil(image, detections)

# -----------------------------
# Helper Functions
# -----------------------------
def parse_data_url(data_url: str) -> Tuple[str, bytes]:
    """Parse data URL menjadi (mime, bytes)"""
    if not data_url.startswith("data:"):
        raise ValueError("Bukan data URL yang valid")
    header, b64data = data_url.split(",", 1)
    mime = header.split(";")[0].split(":", 1)[1]
    raw = base64.b64decode(b64data)
    return mime, raw

def chip(text: str) -> str:
    """Menghasilkan HTML sederhana untuk chip/tag."""
    return f'<span class="chip">{text}</span>'

def image_to_base64(image: np.ndarray, format: str = 'jpg') -> str:
    """Convert image array to base64 string"""
    try:
        if CV2_AVAILABLE:
            # Use OpenCV if available
            _, img_encoded = cv2.imencode(f'.{format}', image)
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            return f"data:image/{format};base64," + img_base64
        else:
            # Fallback to PIL
            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image.astype(np.uint8))
            else:
                pil_image = Image.fromarray(image.astype(np.uint8), mode='L')
            
            # Save to bytes
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format=format.upper())
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
            return f"data:image/{format};base64," + img_base64
    except Exception as e:
        st.error(f"❌ Error encoding image: {str(e)}")
        return ""

def format_detection_time() -> str:
    """Format current time for display"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_detection_stats(detections: List[Dict]) -> Dict:
    """Calculate statistics from detection results"""
    if not detections:
        return {
            "total_objects": 0,
            "unique_foods": 0,
            "avg_confidence": 0.0,
            "highest_confidence": 0.0
        }
    
    total_objects = len(detections)
    unique_foods = len(set(d["nama"] for d in detections))
    avg_confidence = sum(d["confidence"] for d in detections) / total_objects
    highest_confidence = max(d["confidence"] for d in detections)
    
    return {
        "total_objects": total_objects,
        "unique_foods": unique_foods,
        "avg_confidence": round(avg_confidence, 3),
        "highest_confidence": round(highest_confidence, 3)
    }

def save_to_history(detection_results: Dict):
    """Save detection results to session history"""
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    history_entry = {
        "timestamp": detection_results["timestamp"],
        "detections": detection_results["detections"],
        "stats": detection_results["stats"]
    }
    
    st.session_state.detection_history.append(history_entry)
    
    # Keep only last 10 detections
    if len(st.session_state.detection_history) > 10:
        st.session_state.detection_history = st.session_state.detection_history[-10:]

def get_nutritional_analysis(detected_foods):
    """Get nutritional analysis from Groq API"""
    if not GROQ_API_KEY:
        return "⚠️ Groq API key not configured."
    
    makanan_str = ', '.join(list(set(detected_foods)))
    if not makanan_str:
        return "Tidak ada makanan terdeteksi untuk analisis gizi."
    
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
        response.raise_for_status()
        completion = response.json()
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error getting nutritional analysis: {str(e)}"

def create_demo_detection(image):
    """Create demo detection results for testing"""
    import random
    
    # Sample Indonesian food detections with reasonable confidence
    possible_foods = [
        ("nasi goreng", 0.85),
        ("ayam goreng", 0.92),
        ("telur", 0.78),
        ("tempe", 0.73),
        ("tahu", 0.81),
        ("sate", 0.88),
        ("rendang", 0.95),
        ("gado-gado", 0.77)
    ]
    
    # Select 2-4 random foods
    num_foods = random.randint(2, 4)
    selected_foods = random.sample(possible_foods, min(num_foods, len(possible_foods)))
    
    detected_objects = []
    makanan_list = []
    
    img_width, img_height = image.size
    
    for i, (food_name, confidence) in enumerate(selected_foods):
        # Generate random bounding box
        x1 = random.randint(50, img_width // 2)
        y1 = random.randint(50, img_height // 2)
        box_width = random.randint(100, 200)
        box_height = random.randint(80, 150)
        x2 = min(x1 + box_width, img_width - 10)
        y2 = min(y1 + box_height, img_height - 10)
        
        detected_objects.append({
            "nama": food_name,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })
        makanan_list.append(food_name)
    
    # Get nutritional analysis
    gizi_text = get_nutritional_analysis(makanan_list)
    
    # Draw boxes on image using PIL
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    try:
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
    
    # Convert to base64
    img_bytes = io.BytesIO()
    annotated_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return {
        "objects": detected_objects,
        "image": "data:image/jpeg;base64," + img_base64,
        "gizi": gizi_text,
        "annotated_image": np.array(annotated_image)
    }

def perform_local_detection(image: Image.Image):
    """Perform food detection using local YOLO model"""
    try:
        st.session_state.processing = True
        st.session_state.error_message = None
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Preprocess image
        status_text.text("🔄 Preprocessing image...")
        progress_bar.progress(20)
        image_array = preprocess_image(image)
        time.sleep(0.5)
        
        # Step 2: Run detection
        status_text.text("🔍 Detecting food objects...")
        progress_bar.progress(40)
        detections, food_names = detect_food_objects(
            st.session_state.model,
            image_array,
            st.session_state.confidence_threshold
        )
        time.sleep(0.5)
        
        # Step 3: Draw bounding boxes
        status_text.text("🎨 Drawing bounding boxes...")
        progress_bar.progress(60)
        annotated_image = draw_bounding_boxes(image_array, detections)
        time.sleep(0.5)
        
        # Step 4: Get nutritional analysis
        status_text.text("🥗 Analyzing nutrition...")
        progress_bar.progress(80)
        nutritional_info = get_nutritional_analysis(food_names)
        time.sleep(0.5)
        
        # Step 5: Finalize
        status_text.text("✅ Selesai!")
        progress_bar.progress(100)
        
        # Store results
        results = {
            "detections": detections,
            "annotated_image": annotated_image,
            "nutritional_info": nutritional_info,
            "stats": calculate_detection_stats(detections),
            "timestamp": format_detection_time()
        }
        
        st.session_state.detection_results = results
        
        # Save to history
        save_to_history(results)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.processing = False
        st.session_state.success_message = "Deteksi berhasil!"
        st.rerun()
        
    except Exception as e:
        st.session_state.processing = False
        st.session_state.error_message = f"❌ Error during detection: {str(e)}"
        st.error(st.session_state.error_message)

# -----------------------------
# Sidebar Configuration
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ **Pengaturan dan Kontrol**")
    
    # System Status
    st.markdown("### 📊 **Status Sistem**")
    
    # Environment Status
    if not YOLO_AVAILABLE:
        st.error("❌ **YOLO/OpenCV Tidak Tersedia**")
        st.info("🌐 Mode: Demo Only")
        st.markdown("*Environment tidak mendukung OpenCV*")
    elif not CV2_AVAILABLE:
        st.warning("⚠️ **OpenCV Tidak Tersedia**")
        st.info("🖥️ Mode: Local + PIL")
        st.markdown("*Menggunakan PIL untuk image processing*")
    else:
        st.success("✅ **Full Local Mode**")
        st.info("🖥️ Mode: Local + OpenCV")
        st.markdown("*Semua library tersedia*")
    
    st.divider()
    
    # Detection Settings
    st.markdown("### 🔍 **Pengaturan Deteksi**")
    
    # Demo Mode Toggle
    demo_mode = st.checkbox(
        "🎮 **Demo Mode (Offline)**",
        value=False,  # Default ke False agar mode server/local menjadi prioritas
        help="Simulasi deteksi tanpa memerlukan model YOLO - cocok untuk demo dan testing"
    )
    
    # Confidence Threshold
    confidence_threshold = st.slider(
        "📊 **Confidence Threshold**",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence level untuk deteksi objek. Semakin tinggi semakin selektif."
    )
    
    # Update session state
    st.session_state.confidence_threshold = confidence_threshold
    
    st.divider()
    
    # Model Management
    st.markdown("### 🤖 **Model Management**")
    
    if not demo_mode and YOLO_AVAILABLE:
        if st.session_state.model_loaded:
            st.success("✅ **Model Loaded**")
            if os.path.exists(DEFAULT_MODEL_PATH):
                size_mb = os.path.getsize(DEFAULT_MODEL_PATH) / (1024 * 1024)
                st.caption(f"📁 Model: {size_mb:.1f} MB")
            
            if st.button("🔄 Reload Model", type="secondary"):
                st.session_state.model_loaded = False
                st.session_state.model = None
                st.rerun()
        else:
            st.warning("⚠️ **Model Not Loaded**")
            if st.button("🚀 Load YOLO Model", type="primary"):
                with st.spinner("Loading model..."):
                    model = load_yolo_model(st.session_state.model_path)
                    if model:
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                        st.rerun()
                    else:
                        st.error("❌ Gagal load model")
    else:
        st.info("ℹ️ **Demo Mode Aktif**")
        st.caption("Model tidak diperlukan dalam mode demo")
    
    st.divider()
    
    # API Status
    st.markdown("### 🌐 **API Status**")
    
    # Groq API Status
    if GROQ_API_KEY:
        st.success("✅ **Groq API Ready**")
        st.caption("LLM: Groq Llama-4")
    else:
        st.error("❌ **Groq API Key Missing**")
        st.caption("Nutritional analysis tidak akan berfungsi")
    
    st.divider()
    
    # Statistics & History
    st.markdown("### 📈 **Statistik & Riwayat**")
    
    # Detection History
    if st.session_state.detection_history:
        st.caption(f"📊 Total Deteksi: {len(st.session_state.detection_history)}")
        
        if st.button("🗑️ **Clear History", type="secondary"):
            st.session_state.detection_history = []
            st.rerun()
        
        # Show recent detections
        with st.expander("📋 **Riwayat Deteksi Terakhir**"):
            for i, entry in enumerate(reversed(st.session_state.detection_history[-5:]), 1):
                st.markdown(f"**{i}.** {entry['timestamp']}")
                st.caption(f"📦 Objek: {entry['stats']['total_objects']} | 🍽️ Jenis: {entry['stats']['unique_foods']}")
    else:
        st.caption("📭 Belum ada riwayat deteksi")
    
    st.divider()
    
    # App Info
    st.markdown("### ℹ️ **Informasi Aplikasi**")
    
    st.markdown("""
    **DataLens-AI Food Detection**
    
    🤖 **Model:** YOLOv11
    🧠 **LLM:** Groq Llama-4
    🍽️ **Kelas:** 29 Makanan Indonesia
    🖼️ **Format:** JPG/PNG
    ⚡ **Mode:** Local + API Fallback
    """)
    
    st.caption("_Version 1.0 - Streamlit Cloud Ready_")

# -----------------------------
# Main Application
# -----------------------------
inject_custom_styles()

# Header
st.markdown('<h1 class="main-header">🍽️ DataLens-AI Food Detection</h1>', unsafe_allow_html=True)
st.write(
    "Sistem deteksi makanan Indonesia menggunakan **YOLOv11** untuk deteksi objek dan **Groq LLM** "
    "untuk analisis gizi. Upload gambar makanan dan dapatkan informasi gizi lengkap dalam bahasa Indonesia."
)

# Check availability and show appropriate mode in main area
if not YOLO_AVAILABLE:
    st.warning("⚠️ **YOLO/OpenCV Tidak Tersedia** - Menggunakan Demo Mode")
    st.info("💡 Environment Streamlit Cloud tidak mendukung OpenCV. Gunakan demo mode untuk simulasi.")
elif not CV2_AVAILABLE:
    st.warning("⚠️ **OpenCV Tidak Tersedia** - Menggunakan PIL untuk image processing")
    st.success("🖥️ **Local Mode Aktif** - Menggunakan model YOLO lokal dengan PIL")
else:
    st.success("🖥️ **Local Mode Aktif** - Menggunakan model YOLO lokal dengan OpenCV")

# File upload section with status indicator
col1, col2 = st.columns([3, 1])
with col1:
    uploaded = st.file_uploader(
        "📤 **Unggah Gambar Makanan**",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="food_image_uploader",
        help="Upload gambar makanan dalam format JPG atau PNG"
    )

with col2:
    # Show detection readiness status
    if demo_mode:
        st.success("🎮 **Demo Ready**")
        st.caption("Mode: Offline")
    elif st.session_state.model_loaded:
        st.success("🤖 **Model Ready**")
        st.caption("Mode: Local")
    else:
        st.warning("⚠️ **Model Needed**")
        st.caption("Load model di sidebar")

# Handle file preview
if uploaded:
    try:
        uploaded.seek(0)
        image = Image.open(uploaded)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Image preview with info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(image, caption="📷 **Pratinjau Gambar**")
        with col2:
            img_width, img_height = image.size
            st.markdown("### 📊 **Info Gambar**")
            st.metric("Lebar", f"{img_width}px")
            st.metric("Tinggi", f"{img_height}px")
            st.metric("Format", image.format)
            st.metric("Mode", image.mode)
            
    except Exception as e:
        st.error(f"❌ Error loading preview: {str(e)}")
        uploaded = None
else:
    st.info("📷 **Silakan unggah gambar makanan untuk memulai deteksi**")
    st.markdown("""
    **Tips untuk hasil terbaik:**
    - 📸 Gunakan gambar yang jelas dan terang
    - 🍽️ Pastikan makanan terlihat dengan jelas
    - 📏 Hindari gambar yang terlalu kecil atau buram
    - 🎯 Fokus pada makanan yang ingin dideteksi
    """)

# Detection button
if uploaded:
    can_detect = demo_mode or st.session_state.model_loaded
    
    if can_detect:
        if st.button("🔎 Deteksi Gizi", type="primary"):
            with st.spinner("Memproses..."):
                try:
                    uploaded.seek(0)
                    image_bytes = uploaded.read()
                    
                    if not image_bytes:
                        raise ValueError("File is empty or could not be read")
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    if demo_mode:
                        # Use demo detection
                        st.info("🎮 Menggunakan demo mode...")
                        demo_result = create_demo_detection(image)
                        
                        # Convert demo result to expected format
                        results = {
                            "detections": demo_result["objects"],
                            "annotated_image": demo_result["annotated_image"],
                            "nutritional_info": demo_result["gizi"],
                            "stats": calculate_detection_stats(demo_result["objects"]),
                            "timestamp": format_detection_time()
                        }
                        
                        st.session_state.detection_results = results
                        st.session_state.success_message = "Demo deteksi berhasil!"
                        st.rerun()
                    else:
                        # Use local detection
                        perform_local_detection(image)
                        
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
    else:
        st.warning("⚠️ Load model terlebih dahulu atau gunakan Demo Mode")

# Display results
if st.session_state.detection_results:
    results = st.session_state.detection_results
    
    st.subheader("🎯 Hasil Deteksi")
    
    # Success message
    if st.session_state.success_message:
        st.success(st.session_state.success_message)
    
    # Statistics
    stats = results["stats"]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Objek", stats["total_objects"])
    with col2:
        st.metric("Jenis Makanan", stats["unique_foods"])
    with col3:
        st.metric("Rata-rata Confidence", stats["avg_confidence"])
    with col4:
        st.metric("Confidence Tertinggi", stats["highest_confidence"])
    
    # Display timestamp
    st.caption(f"🕒 Deteksi pada: {results['timestamp']}")
    
    # Results columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📷 Gambar dengan Deteksi")
        
        # Convert and display annotated image
        annotated_image = results["annotated_image"]
        
        # Handle image display based on whether OpenCV is available
        if CV2_AVAILABLE and len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
            # Convert BGR to RGB if OpenCV was used
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Hasil Deteksi dengan Bounding Box")
        else:
            # Display as-is if PIL was used (already RGB)
            st.image(annotated_image, caption="Hasil Deteksi dengan Bounding Box")
        
        # Download button
        img_base64 = image_to_base64(annotated_image)
        if img_base64:
            st.markdown("### 💾 Download")
            st.markdown(
                f'<a href="{img_base64}" download="detected_food.jpg" class="download-btn">📥 Unduh Gambar Hasil</a>',
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown("### 📦 Detail Deteksi")
        
        detections = results["detections"]
        if detections:
            # Create detection table
            detection_data = []
            for i, det in enumerate(detections, 1):
                detection_data.append({
                    "No": i,
                    "Makanan": det["nama"],
                    "Confidence": f"{det['confidence']:.3f}",
                    "Posisi": f"({det['bbox'][0]}, {det['bbox'][1]})"
                })
            
            df = pd.DataFrame(detection_data)
            st.dataframe(df)
            
            # Food tags
            st.markdown("### 🏷️ Makanan Terdeteksi")
            food_tags = ""
            for food in set(det["nama"] for det in detections):
                food_tags += f'<span class="chip">{food}</span>'
            st.markdown(food_tags, unsafe_allow_html=True)
        else:
            st.info("Tidak ada objek terdeteksi dengan confidence threshold saat ini.")
    
    # Nutritional Information
    st.markdown("### 🥗 Analisis Gizi")
    nutritional_info = results["nutritional_info"]
    
    if nutritional_info:
        # Check if response contains table format
        if "|" in nutritional_info and "-" in nutritional_info:
            st.markdown(nutritional_info)
        else:
            st.text_area("Informasi Gizi", nutritional_info, height=200)
    else:
        st.info("Tidak ada informasi gizi tersedia.")

# Footer
st.markdown("---")
st.markdown("**Mode:** Local Detection | **Model:** YOLOv11 | **LLM:** Groq Llama-4")
