import os, torch, time, cv2, numpy as np, gradio as gr

# ==============================================================
# 🚀 Import YOLO
# ==============================================================
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
model.to(device)
print("✅ Model loaded successfully!")
print("Model class names:", model.names)


# ==============================================================
# 🧠 Inference Function
# ==============================================================
def detect(image):
    """
    Run YOLO inference on the uploaded image and classify tomato quality.
    """
    try:
        start_time = time.time()
        
        # Run YOLO prediction
        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            augment=True,
            verbose=False,
            device=device
        )

        annotated = results[0].plot()
        detected_classes = [model.names[int(box.cls[0])].strip().lower() for box in results[0].boxes]

        # Quality classification logic
        conclusion = "No Tomato Detected"
        if any("damaged" in c for c in detected_classes):
            conclusion = "Damaged 🍂"
        elif any("unripe" in c for c in detected_classes):
            conclusion = "Unripe 🍏"
        elif any("ripe" in c for c in detected_classes):
            conclusion = "Ripe 🍅"

        fps = 1.0 / (time.time() - start_time)
        print(f"🕒 FPS: {fps:.1f} | Detections: {len(results[0].boxes)} | Final: {conclusion}")

        return annotated, conclusion

    except Exception as e:
        print("❌ Error:", e)
        return image, f"Error: {str(e)}"


# ==============================================================
# 🖥️ Gradio Interface
# ==============================================================
interface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy", label="📷 Live Webcam Feed", sources=["webcam"], streaming=True),
    outputs=[
        gr.Image(type="numpy", label="Detected Image"),
        gr.Textbox(label="Conclusion")
    ],
    title="Tomato Quality Detector 🍅",
    description="Point your webcam at a tomato for real-time quality detection (Ripe / Unripe / Damaged).",
    live=True
)

if __name__ == "__main__":
    interface.launch()
