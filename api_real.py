from ultralytics import YOLO
from flask import Flask, request, jsonify
import numpy as np
import os
import cv2
from flask_cors import CORS
import requests
import io
import base64

app = Flask(__name__)
CORS(app)

# Groq API Configuration
KEY = "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"
MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Load YOLO model
model_path = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")
print(f"🔍 Loading model from: {model_path}")

try:
    modelyolo = YOLO(model_path)
    print(f"✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    modelyolo = None

class_names = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

def preprocess_image(image_data):
    """Preprocess image data for YOLO"""
    try:
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        raise ValueError(f"Image preprocessing error: {e}")

def draw_boxes(image, results):
    """Draw bounding boxes on image"""
    if results is None or len(results.boxes) == 0:
        return image
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        conf = float(box.conf[0])
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return image

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": modelyolo is not None,
        "model_path": model_path,
        "classes": len(class_names),
        "class_names": class_names
    })

@app.route("/detect-gizi", methods=["POST"])
def detect_gizi():
    """Main detection endpoint with real YOLO model"""
    try:
        # Check if model is loaded
        if modelyolo is None:
            return jsonify({"error": "YOLO model not loaded. Check model path: " + model_path}), 500
        
        # Get image file
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400
        
        # Process image
        image_data = image_file.read()
        image = preprocess_image(image_data)
        
        # Run YOLO detection
        print(f"🔍 Running detection on image shape: {image.shape}")
        results = modelyolo(image)[0]
        
        detected_objects = []
        makanan_list = []
        
        print(f"📊 Found {len(results.boxes)} objects")
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            conf = float(box.conf[0])
            
            print(f"🍽️ Detected: {label} with confidence {conf:.3f}")
            
            makanan_list.append(label)
            detected_objects.append({
                "nama": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
        
        # Generate nutrition info using Groq LLM
        gizi_text = ""
        if makanan_list:
            try:
                makanan_str = ', '.join(list(set(makanan_list)))
                prompt = f"Berikan informasi kandungan gizi dari makanan berikut: {makanan_str}. Jawab dalam bentuk tabel dan bahasa Indonesia."
                
                print(f"🤖 Getting nutrition info for: {makanan_str}")
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": "Kamu adalah asisten gizi makanan Indonesia."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                    },
                    timeout=30,
                )
                
                if response.status_code == 200:
                    completion = response.json()
                    gizi_text = completion["choices"][0]["message"]["content"]
                    print("✅ Nutrition info generated successfully")
                else:
                    gizi_text = f"Error getting nutrition info: {response.status_code}"
                    print(f"❌ Nutrition API error: {response.status_code}")
                    
            except Exception as e:
                gizi_text = f"Error getting nutrition info: {str(e)}"
                print(f"❌ Nutrition error: {e}")
        else:
            gizi_text = "Tidak ada makanan yang terdeteksi"
            print("⚠️ No food detected")
        
        # Draw boxes and encode image
        boxed_image = draw_boxes(image.copy(), results)
        _, img_encoded = cv2.imencode('.jpg', boxed_image)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        result = {
            "success": True,
            "objects": detected_objects,
            "image": "data:image/jpeg;base64," + img_base64,
            "gizi": gizi_text,
            "detected_foods": list(set(makanan_list)),
            "total_detections": len(detected_objects)
        }
        
        print(f"✅ Detection completed. Found {len(detected_objects)} objects")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in detect_gizi: {e}")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

if __name__ == "__main__":
    print("🚀 Starting Food Detection API Server with Real YOLO Model...")
    print(f"📍 Model path: {model_path}")
    print(f"🔗 Server will run on http://localhost:5000")
    print(f"🍽️ Supporting {len(class_names)} Indonesian food classes")
    app.run(host="0.0.0.0", port=5000, debug=True)