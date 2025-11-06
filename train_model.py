from ultralytics import YOLO
import os

def main():
    print("🚀 Starting YOLO training for Indonesian food detection...")
    
    # Check dataset
    data_yaml = "dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset config not found: {data_yaml}")
        return
    
    # Initialize model
    model_name = "yolo11n.pt"  # Use YOLOv11 nano (smaller, faster)
    print(f"📥 Loading base model: {model_name}")
    
    try:
        model = YOLO(model_name)
        
        # Training parameters
        print("🏋️ Starting training...")
        print(f"📊 Dataset: {data_yaml}")
        print(f"🔄 Epochs: 25")
        print(f"🖼️ Image size: 224")
        
        # Train model
        results = model.train(
            data=data_yaml, 
            epochs=25, 
            imgsz=224,
            batch=16,
            name='train2',  # This will create runs/detect/train2/
            save=True,
            plots=True,
            verbose=True
        )
        
        print("✅ Training completed!")
        print(f"📁 Model saved to: runs/detect/train2/weights/best.pt")
        
        # Validate model
        print("🔍 Validating model...")
        model = YOLO("runs/detect/train2/weights/best.pt")
        
        metrics = model.val(
            data="dataset/data.yaml", 
            split="val", 
            imgsz=224, 
            batch=16, 
            conf=0.001, 
            iou=0.65, 
            plots=True
        )
        
        print("✅ Validation completed!")
        print(f"📈 mAP50: {metrics.box.map50:.4f}")
        print(f"📈 mAP50-95: {metrics.box.map:.4f}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return

if __name__ == "__main__":
    main()