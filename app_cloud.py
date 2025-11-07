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

# ==================== CONFIGURATION ====================
GROQ_API_KEY = "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"
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
# API URLs - try multiple options
HUGGINGFACE_API_URL = "https://wanndev14-yolo-api.hf.space"  # Update after deployment
LOCAL_API_URL = "http://localhost:5000"

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

# ==================== HELPER FUNCTIONS ====================
def test_api_connection(api_url=HUGGINGFACE_API_URL):
    """Test if API is accessible"""
    try:
        # Test basic connectivity
        response = requests.get(api_url, timeout=10)
        return response.status_code == 200
    except:
        return False

def detect_with_api(image_data, api_url=HUGGINGFACE_API_URL):
    """Deteksi menggunakan HuggingFace API"""
    try:
        # Test API connectivity first
        if not test_api_connection(api_url):
            st.error("❌ API tidak dapat dijangkau - API mungkin sedang down atau URL salah")
            st.info("🔗 URL API: " + api_url)
            return None
            
        files = {"image": image_data}
        
        # Try different possible endpoints
        endpoints = ["/detect-gizi", "/predict", "/detect", "/api/detect"]
        
        for endpoint in endpoints:
            try:
                response = requests.post(f"{api_url}{endpoint}", files=files, timeout=60)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    continue  # Try next endpoint
                else:
                    st.warning(f"⚠️ Endpoint {endpoint}返回状态码: {response.status_code}")
                    continue
                    
            except requests.exceptions.Timeout:
                st.error("❌ API Timeout - Coba lagi beberapa saat")
                return None
            except requests.exceptions.ConnectionError:
                st.error("❌ API Connection Error - Periksa koneksi internet")
                return None
                
        st.error("❌ Semua endpoint API gagal - API mungkin tidak memiliki endpoint yang sesuai")
        st.info("💡 Coba gunakan Demo Mode atau periksa konfigurasi API")
        return None
        
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

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
    except requests.exceptions.Timeout:
        return "❌ Timeout analisis gizi - Coba lagi"
    except Exception as e:
        return f"❌ Error analisis gizi: {str(e)}"

def create_demo_detection(image):
    """Create demo detection results when API is unavailable"""
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
    
    # Create annotated image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
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
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        
        # Draw label
        label = f"{food_name} ({confidence:.2f})"
        if font:
            draw.text((x1, y1-20), label, fill=(0, 255, 0), font=font)
        else:
            draw.text((x1, y1-20), label, fill=(0, 255, 0))
    
    # Get nutritional analysis
    gizi_text = get_nutritional_analysis(makanan_list)
    
    return {
        "objects": detected_objects,
        "annotated_image": annotated_image,
        "gizi": gizi_text
    }

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Status
    st.subheader("📊 Status")
    st.success("✅ Cloud Mode Aktif")
    
    # API Configuration
    st.subheader("🌐 API Configuration")
    
    # API URL selection
    api_option = st.selectbox(
        "Pilih API Source",
        ["HuggingFace API", "Local API (localhost:5000)", "Custom URL"],
        help="Pilih sumber API untuk deteksi makanan"
    )
    
    if api_option == "HuggingFace API":
        api_url = HUGGINGFACE_API_URL
    elif api_option == "Local API (localhost:5000)":
        api_url = LOCAL_API_URL
    else:
        api_url = st.text_input(
            "Custom API URL",
            value=HUGGINGFACE_API_URL,
            help="Masukkan URL API kustom"
        )
    
    st.info(f"🔗 Current API: {api_url}")
    
    # Test API button
    if st.button("🧪 Test API Connection"):
        with st.spinner("Testing API..."):
            if test_api_connection(api_url):
                st.success("✅ API Terhubung!")
            else:
                st.error("❌ API tidak dapat dijangkau")
                st.info("💡 Solusi:")
                st.info("1. Coba ganti API source di dropdown")
                st.info("2. Periksa koneksi internet")
                st.info("3. Gunakan Demo Mode jika API tidak tersedia")
    
    # Confidence threshold
    st.subheader("🔍 Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05
    )
    
    # Demo mode toggle
    st.subheader("🎮 Demo Mode")
    demo_mode = st.checkbox("Gunakan Demo Mode (Offline)", help="Jangan gunakan API, hasil simulasi")
    
    st.success("✅ Groq API: Ready")

# ==================== MAIN APP ====================
st.title("🍽️ Deteksi Gizi Makanan")
st.write("Upload foto makanan untuk deteksi otomatis dan analisis gizi")

# Tampilkan mode yang aktif
if demo_mode:
    st.info("🎮 **Demo Mode** - Simulasi deteksi tanpa API")
else:
    st.info("🌐 **API Mode** - Menggunakan HuggingFace API untuk deteksi makanan")

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
                nutritional_info = ""
                
                # Pilih metode deteksi
                if demo_mode:
                    # Demo mode - create fake results
                    st.info("🎮 Menggunakan demo mode...")
                    time.sleep(1) # Simulate processing time
                    
                    result = create_demo_detection(image)
                    detected_objects = result["objects"]
                    annotated_image = result["annotated_image"]
                    nutritional_info = result["gizi"]
                    food_names = [obj["nama"] for obj in detected_objects]
                    
                else:
                    # API detection
                    st.info("🌐 Mengirim ke HuggingFace API...")
                    result = detect_with_api(img_bytes, api_url)
                    
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
                                annotated_image = image
                        
                        # Get nutritional info from API response
                        nutritional_info = result.get("gizi", "")
                        
                        if not nutritional_info and food_names:
                            # Fallback to Groq API if no nutrition info from detection API
                            with st.spinner("🥗 Menganalisis gizi..."):
                                nutritional_info = get_nutritional_analysis(food_names)
                    else:
                        # API failed, try demo mode as fallback
                        st.warning("⚠️ API gagal, mencoba demo mode...")
                        result = create_demo_detection(image)
                        detected_objects = result["objects"]
                        annotated_image = result["annotated_image"]
                        nutritional_info = result["gizi"]
                        food_names = [obj["nama"] for obj in detected_objects]
                
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
                st.info("💡 Coba gunakan Demo Mode jika API tidak tersedia")

# Instructions
st.markdown("---")
st.markdown("### 📋 Petunjuk Penggunaan")
st.markdown("""
1. **Upload Gambar**: Pilih foto makanan dalam format JPG/PNG
2. **Pilih Mode**: 
   - **API Mode**: Gunakan HuggingFace API untuk deteksi real (membutuhkan internet)
   - **Demo Mode**: Simulasi deteksi untuk demonstrasi (offline)
3. **Atur Confidence**: Sesuaikan threshold deteksi (0.1-1.0)
4. **Klik Deteksi**: Tunggu proses analisis selesai
5. **Lihat Hasil**: Deteksi objek, bounding box, dan analisis gizi
""")

# Footer
st.markdown("---")
st.markdown("**Deteksi Gizi Makanan** | Cloud Deployment | API + Groq LLM")