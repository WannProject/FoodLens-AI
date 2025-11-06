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
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    GROQ_MODEL = st.secrets.get("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Class names - Synchronized with dataset/data.yaml
CLASS_NAMES = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

# -----------------------------
# Helper Functions
# -----------------------------
def parse_data_url(data_url: str) -> Tuple[str, bytes]:
    """Mem-parse data URL "data:<mime>;base64,<payload>" menjadi (mime, bytes)"""
    if not data_url.startswith("data:"):
        raise ValueError("Bukan data URL yang valid")
    header, b64data = data_url.split(",", 1)
    mime = header.split(";")[0].split(":", 1)[1]
    raw = base64.b64decode(b64data)
    return mime, raw

def chip(text: str) -> str:
    """Menghasilkan HTML sederhana untuk chip/tag."""
    return f'<span class="chip">{text}</span>'

def inject_style():
    st.markdown("""
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
    """, unsafe_allow_html=True)

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

def detect_with_api(image_data, api_url="http://localhost:5000"):
    """Try to detect using external API"""
    try:
        files = {"image": image_data}
        response = requests.post(f"{api_url}/detect-gizi", files=files, timeout=90)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def create_demo_detection(image):
    """Create demo detection results for testing"""
    # Generate random but realistic detections
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
    
    # Draw boxes on image
    image_array = np.array(image)
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{obj['nama']} ({obj['confidence']:.2f})"
        cv2.putText(image_array, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    # Convert to base64
    _, img_encoded = cv2.imencode('.jpg', image_array)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
    
    return {
        "objects": detected_objects,
        "image": "data:image/jpeg;base64," + img_base64,
        "gizi": gizi_text,
        "annotated_image": image_array
    }

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Mode selection
    mode = st.radio(
        "Mode Deteksi",
        options=["API Mode", "Demo Mode"],
        help="API: Gunakan server API eksternal. Demo: Mode simulasi untuk testing."
    )
    
    if mode == "Demo Mode":
        st.info("🎭 Demo Mode - Menampilkan hasil simulasi")
        st.caption("Tidak memerlukan model file untuk testing.")
    else:
        st.info("🔌 API Mode - Menghubungkan ke server API")
        st.caption("Pastikan API server berjalan di http://localhost:5000")
        
        api_url = st.text_input(
            "API URL",
            value="http://localhost:5000",
            help="URL untuk API server deteksi makanan"
        )
    
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

# Reset file pointer if file exists
if uploaded:
    uploaded.seek(0)

with col_preview:
    if uploaded:
        try:
            # Reset file pointer to beginning
            uploaded.seek(0)
            image = Image.open(uploaded)
            # Convert image to RGB if it's not already (fixes some image format issues)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            st.image(image, caption="Pratinjau Gambar", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.stop()

st.write("")
st.write("")
detect_btn = st.button("🔎 Deteksi Gizi", type="primary", disabled=not uploaded)

# -----------------------------
# Hasil
# -----------------------------
if detect_btn and uploaded:
    try:
        with st.spinner("Memproses..."):
            # Read image directly from uploaded file
            uploaded.seek(0)  # Reset file pointer
            image_bytes = uploaded.read()
            try:
                image = Image.open(io.BytesIO(image_bytes))
                # Convert image to RGB if it's not already
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.stop()
            
            if mode == "Demo Mode":
                # Demo simulation
                st.success("🎭 Demo Mode - Simulating detection results...")
                result = create_demo_detection(image)
                
            else:
                # API Mode
                st.info("🔌 Menghubungkan ke API server...")
                
                # Try API detection
                result = detect_with_api(io.BytesIO(image_bytes), api_url)
                
                if result is None:
                    st.error("❌ Tidak dapat terhubung ke API server.")
                    st.info("Pastikan API server berjalan dan coba lagi, atau gunakan Demo Mode.")
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
                            # Convert image to RGB if it's not already
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            st.image(image, caption="Gambar dengan Bounding Box", use_container_width=True)
                            st.download_button(
                                label="Unduh Gambar Hasil",
                                data=img_bytes,
                                file_name="hasil_deteksi.jpg",
                                mime=mime,
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
                    st.dataframe(df, hide_index=True)
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
    st.markdown("**Mode:** API Detection | **Model:** YOLOv11 | **LLM:** Groq Llama-3.1-70B")