# -*- coding: utf-8 -*-
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

KEY = "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"
MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

model_path = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")
modelyolo = YOLO(model_path)

class_names = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

def preprocess_image(image_data):
    image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def draw_boxes(image, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = class_names[cls]
        conf = float(box.conf[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return image

@app.route("/detect-gizi", methods=["POST"])
def detect_gizi():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    image_data = image_file.read()
    image = preprocess_image(image_data)

    results = modelyolo(image)[0]

    detected_objects = []
    makanan_list = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = class_names[cls]
        conf = float(box.conf[0])
        makanan_list.append(label)
        detected_objects.append({
            "nama": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    # Prompt makanan yang terdeteksi ke LLM untuk gizi
    makanan_str = ', '.join(list(set(makanan_list)))  # Unique
    prompt = f"Berikan informasi kandungan gizi dari makanan berikut: {makanan_str}. Jawab dalam bentuk tabel dan bahasa Indonesia."

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
    completion = response.json()
    gizi_text = completion["choices"][0]["message"]["content"]

    boxed_image = draw_boxes(image.copy(), results)
    _, img_encoded = cv2.imencode('.jpg', boxed_image)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    return jsonify({
        "objects": detected_objects,
        "image": "data:image/jpeg;base64," + img_base64,
        "gizi": gizi_text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
