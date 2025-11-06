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
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = st.secrets.get("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

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
    model_path = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"Model file not found: {model_path}")
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
        completion = response.json()
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error getting nutritional analysis: {str(e)}"

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
        st.error(f"Error during detection: {str(e)}")
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
        options=["Local (Built-in)", "API Eksternal"],
        help="Local: Tanpa perlu API server. API: Gunakan server Flask terpisah."
    )
    
    if mode == "API Eksternal":
        api_base = st.text_input(
            "API Base URL",
            value="http://localhost:5000",
            help="Isi dengan alamat API Flask kamu."
        )
        endpoint_path = st.text_input(
            "Endpoint Deteksi",
            value="/detect-gizi",
            help="Path endpoint pada API Flask."
        )
        st.caption("Pastikan API sudah berjalan. Contoh: app Flask berjalan di http://localhost:5000.")
    else:
        st.success("✅ Mode Local - Tidak perlu API server")
        st.caption("Deteksi dilakukan langsung di browser menggunakan YOLO model.")
    
    conf_filter = st.slider(
        "Filter Confidence (untuk tampilan list, bukan mempengaruhi bounding box dari server)",
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

# Status indicator
if mode == "Local (Built-in)":
    model = load_model()
    if model:
        st.success("✅ YOLO Model loaded successfully")
    else:
        st.error("❌ YOLO Model not found - Please check model file")

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
        st.image(uploaded, caption="Pratinjau Gambar", use_container_width=True)

st.write("")
st.write("")
detect_btn = st.button("🔎 Deteksi Gizi", type="primary", use_container_width=True, disabled=not uploaded)

# -----------------------------
# Hasil
# -----------------------------
if detect_btn and uploaded:
    try:
        with st.spinner("Memproses..."):
            if mode == "Local (Built-in)":
                # Local detection
                image = Image.open(uploaded)
                result = detect_food_locally(image)
                
                if result is None:
                    st.error("Gagal melakukan deteksi lokal. Pastikan model file ada.")
                    st.stop()
                    
            else:
                # API detection (original code)
                api_url = api_base.rstrip("/") + "/" + endpoint_path.lstrip("/")
                file_bytes = uploaded.read()
                mime = uploaded.type or "image/jpeg"
                
                def post_image(api_url: str, file_name: str, file_bytes: bytes, mime: str, timeout: int = 60) -> Dict:
                    files = {
                        "image": (file_name, file_bytes, mime or "application/octet-stream")
                    }
                    resp = requests.post(api_url, files=files, timeout=timeout)
                    resp.raise_for_status()
                    return resp.json()
                
                result = post_image(api_url, uploaded.name, file_bytes, mime, timeout=90)

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
                        st.image(image, caption="Gambar dengan Bounding Box", use_container_width=True)
                        st.download_button(
                            label="Unduh Gambar Hasil",
                            data=img_bytes,
                            file_name="hasil_deteksi.jpg",
                            mime=mime,
                            use_container_width=True
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
                    use_container_width=True,
                    hide_index=True
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
                
        st.toast("Selesai memproses ✅", icon="✅")
        
    except requests.exceptions.ConnectionError:
        st.error(f"Gagal terhubung ke API. Pastikan API berjalan dan URL benar.")
    except requests.exceptions.Timeout:
        st.error("Permintaan ke API melebihi batas waktu (timeout). Coba lagi.")
    except requests.HTTPError as he:
        try:
            err_json = he.response.json()
        except Exception:
            err_json = he.response.text
        st.error(f"API mengembalikan error {he.response.status_code}: {err_json}")
    except Exception as e:
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
if mode == "Local (Built-in)":
    st.markdown("**Mode:** Local Detection | **Model:** YOLOv11 | **LLM:** Groq Llama-4-Maverick")
else:
    st.markdown("**Mode:** API Eksternal | **Endpoint:** Custom API Server")