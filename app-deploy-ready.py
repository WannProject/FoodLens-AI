import streamlit as st
import requests
import base64
import io
from PIL import Image
import pandas as pd
from typing import Tuple, List, Dict
import numpy as np
import cv2
import os
from ultralytics import YOLO
import tempfile

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(
    page_title="Deteksi Gizi Makanan",
    page_icon="🍽️",
    layout="wide",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "UI Streamlit untuk Deteksi Gizi Makanan (YOLO + LLM)."
    }
)

# -----------------------------
# Configuration
# -----------------------------
# Get API key from environment variable for deployment
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-70b-versatile"

# Class names
CLASS_NAMES = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

# -----------------------------
# Model Loading (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    """Load YOLO model with caching"""
    # Try multiple possible model paths
    model_paths = [
        "best.pt",
        "runs/detect/train2/weights/best.pt",
        "./runs/detect/train2/weights/best.pt"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                return YOLO(model_path)
            except Exception as e:
                st.warning(f"Failed to load model from {model_path}: {e}")
                continue
    
    st.error("❌ YOLO Model not found! Please upload model file or check path.")
    st.info("Model file should be named 'best.pt' and placed in the root directory.")
    return None

def preprocess_image(image):
    """Preprocess image for YOLO detection"""
    if isinstance(image, Image.Image):
        # Convert PIL to numpy array
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("Unsupported image format")

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
    if not GROQ_API_KEY:
        return "⚠️ Groq API key not configured. Please set GROQ_API_KEY environment variable."
    
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

def detect_food_locally(image):
    """Local food detection function (replaces API call)"""
    try:
        # Load model
        model = load_model()
        if model is None:
            return None
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Run YOLO detection
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
        
        # Convert to base64 for consistency with original app
        _, img_encoded = cv2.imencode('.jpg', boxed_image)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        return {
            "objects": detected_objects,
            "image": "data:image/jpeg;base64," + img_base64,
            "gizi": gizi_text,
            "annotated_image": boxed_image
        }
        
    except Exception as e:
        st.error(f"❌ Error during detection: {str(e)}")
        return None

# -----------------------------
# Helper Functions (Original)
# -----------------------------
def parse_data_url(data_url: str) -> Tuple[str, bytes]:
    """
    Mem-parse data URL "data:<mime>;base64,<payload>" menjadi (mime, bytes)
    """
    if not data_url.startswith("data:"):
        raise ValueError("Bukan data URL yang valid")
    header, b64data = data_url.split(",", 1)
    mime = header.split(";")[0].split(":", 1)[1]
    raw = base64.b64decode(b64data)
    return mime, raw

def chip(text: str) -> str:
    """
    Menghasilkan HTML sederhana untuk chip/tag.
    """
    return f"""
    <span class="chip">{text}</span>
    """

def inject_style():
    st.markdown(
        """
        <style>
        .chip {
            display: inline-block;
            padding: 6px 12px;
            margin: 4px 6px 0 0;
            border-radius: 16px;
            background: #EEF2FF;
            color: #3730A3;
            font-size: 12px;
            border: 1px solid #E0E7FF;
            white-space: nowrap;
        }
        .metric-card {
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            background: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Mode selection
    mode = st.radio(
        "Mode Deteksi",
        options=["Local (Built-in)", "Demo Mode"],
        help="Local: Gunakan model YOLO langsung. Demo: Mode simulasi."
    )
    
    if mode == "Demo Mode":
        st.info("🎭 Demo Mode - Menampilkan hasil simulasi")
        st.caption("Tidak memerlukan model file untuk testing.")
    else:
        # Check model status
        model = load_model()
        if model:
            st.success("✅ YOLO Model loaded successfully")
        else:
            st.error("❌ YOLO Model not found")
            st.warning("Please upload 'best.pt' model file")
            
            # Model upload option
            uploaded_model = st.file_uploader(
                "Upload YOLO Model (.pt)",
                type=["pt"],
                help="Upload your trained YOLO model file"
            )
            
            if uploaded_model:
                try:
                    # Save uploaded model
                    with open("best.pt", "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    st.success("✅ Model uploaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to upload model: {e}")
    
    # API Key configuration
    if not GROQ_API_KEY:
        st.warning("⚠️ Groq API Key not configured")
        api_key_input = st.text_input(
            "Enter Groq API Key",
            type="password",
            help="Get your API key from https://console.groq.com/"
        )
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            st.success("✅ API Key configured!")
            st.rerun()
    else:
        st.success("✅ Groq API Key configured")
    
    conf_filter = st.slider(
        "Filter Confidence",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05
    )
    render_markdown_table = st.toggle(
        "Render Tabel Gizi sebagai Markdown",
        value=True,
        help="Jika dimatikan, akan ditampilkan sebagai teks apa adanya."
    )

inject_style()

# -----------------------------
# Header
# -----------------------------
st.title("🍽️ Deteksi Gizi Makanan")
st.write(
    "Unggah foto makanan, sistem akan mendeteksi objek (makanan) dengan YOLO dan meminta LLM untuk "
    "menyusun tabel kandungan gizi. Hasil deteksi divisualisasikan dengan bounding box."
)

# -----------------------------
# Upload & Action
# -----------------------------
uploaded = st.file_uploader(
    "Unggah Gambar (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

col_preview, col_action = st.columns([3, 2], vertical_alignment="bottom")

with col_preview:
    if uploaded:
        st.image(uploaded, caption="Pratinjau Gambar", use_column_width=True)

st.write("")
st.write("")
detect_btn = st.button("🔎 Deteksi Gizi", type="primary", disabled=not uploaded)

# -----------------------------
# Hasil
# -----------------------------
if detect_btn and uploaded:
    try:
        with st.spinner("Memproses..."):
            if mode == "Demo Mode":
                # Demo simulation
                st.success("🎭 Demo Mode - Simulating detection results...")
                
                # Simulate detection results
                demo_objects = [
                    {"nama": "nasi goreng", "confidence": 0.85, "bbox": [100, 100, 300, 200]},
                    {"nama": "ayam goreng", "confidence": 0.92, "bbox": [350, 150, 500, 300]}
                ]
                
                demo_gizi = """
| Makanan | Kalori | Protein | Karbohidrat | Lemak |
|---------|--------|---------|-------------|-------|
| Nasi Goreng | 250 kcal | 5g | 45g | 8g |
| Ayam Goreng | 280 kcal | 25g | 0g | 18g |
                
**Total Estimasi:** 530 kcal
                """
                
                # Load original image for annotation
                image = Image.open(uploaded)
                image_array = np.array(image)
                
                # Draw simple demo boxes
                for obj in demo_objects:
                    x1, y1, x2, y2 = obj["bbox"]
                    cv2.rectangle(image_array, (x1, y1), (x2, y2), (0,255,0), 2)
                    text = f"{obj['nama']} ({obj['confidence']:.2f})"
                    cv2.putText(image_array, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                result = {
                    "objects": demo_objects,
                    "image": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpg', image_array)[1].tobytes()).decode(),
                    "gizi": demo_gizi,
                    "annotated_image": image_array
                }
                
            else:
                # Local detection
                image = Image.open(uploaded)
                result = detect_food_locally(image)
                
                if result is None:
                    st.error("❌ Gagal melakukan deteksi. Pastikan model file sudah diupload.")
                    st.stop()

        # Validasi respon
        if not isinstance(result, dict):
            st.error("Format respon tidak sesuai.")
        else:
            # Gambar beranotasi
            img_data_url = result.get("image")
            objects = result.get("objects", [])
            gizi_text = result.get("gizi", "")

            # Bagian atas: Gambar & Ringkasan
            left, right = st.columns([3, 2], gap="large")

            with left:
                st.subheader("📷 Hasil Deteksi")
                if img_data_url:
                    try:
                        mime, img_bytes = parse_data_url(img_data_url)
                        image = Image.open(io.BytesIO(img_bytes))
                        st.image(image, caption="Gambar dengan Bounding Box", use_column_width=True)
                        st.download_button(
                            label="Unduh Gambar Hasil",
                            data=img_bytes,
                            file_name="hasil_deteksi.jpg",
                            mime=mime
                        )
                    except Exception as e:
                        st.warning(f"Gagal menampilkan gambar beranotasi: {e}")
                else:
                    st.info("Tidak ada gambar beranotasi yang dikembalikan.")

            # Detail objek
            st.subheader("📦 Detail Objek Terdeteksi")
            # Filter berdasarkan threshold untuk tampilan
            filtered = [o for o in objects if float(o.get("confidence", 0.0)) >= conf_filter]
            if filtered:
                # Bentuk tabel data
                def bbox_area(b):
                    try:
                        x1, y1, x2, y2 = b
                        return max(0, x2 - x1) * max(0, y2 - y1)
                    except Exception:
                        return None

                rows = []
                for o in filtered:
                    nama = o.get("nama", "-")
                    conf = float(o.get("confidence", 0.0))
                    bbox = o.get("bbox", [None, None, None, None])
                    rows.append({
                        "Nama": nama,
                        "Confidence": round(conf, 4),
                        "BBox": bbox,
                        "Luas (px^2)": bbox_area(bbox)
                    })
                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("Tidak ada objek yang memenuhi filter confidence saat ini.")

            # Tabel gizi (Markdown dari LLM)
            st.subheader("🥗 Kandungan Gizi")
            if gizi_text:
                if render_markdown_table:
                    st.markdown(gizi_text)
                else:
                    st.text(gizi_text)
            else:
                st.info("Tidak ada analisis gizi yang dikembalikan.")
                
        st.toast("✅ Selesai memproses", icon="✅")
        
    except Exception as e:
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
if mode == "Demo Mode":
    st.markdown("**Mode:** Demo Simulation | **Status:** Ready for Testing")
else:
    st.markdown("**Mode:** Local Detection | **Model:** YOLOv11 | **LLM:** Groq Llama-3.1-70B")