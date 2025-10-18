from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import time
import torch

app = Flask(__name__)

# 🚀 Auto-select GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# 🧠 Load YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
model.to(device)
print("✅ Model loaded successfully!")
print("Model class names:", model.names)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    try:
        start_time = time.time()

        # Get base64 image from frontend
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image data received'}), 400

        # Decode base64 → NumPy frame
        img_bytes = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return jsonify({'error': 'Invalid frame'}), 400

        # 🧩 Improved YOLO prediction
        results = model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            augment=True,
            verbose=False
        )

        # Annotate image
        annotated = results[0].plot()

        # Collect detected class labels (normalized to lowercase)
        detected_classes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls].strip().lower()  # normalize to lowercase
            detected_classes.append(label)
            print(f"Detected: {label} ({conf:.2f})")

        # ✅ Decision hierarchy (case-insensitive)
        conclusion = "No Tomato Detected"
        color = (200, 200, 200)

        if any("damaged" in c for c in detected_classes):
            conclusion = "Damaged"
            color = (0, 0, 255)
        elif any("unripe" in c for c in detected_classes):
            conclusion = "Unripe"
            color = (0, 255, 255)
        elif any("ripe" in c for c in detected_classes):
            conclusion = "Ripe"
            color = (0, 255, 0)

        # 🏷️ Draw conclusion text on the frame
        h, w, _ = frame.shape
        cv2.putText(
            annotated,
            f"Conclusion: {conclusion}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA
        )

        # Encode annotated frame for display
        _, buffer = cv2.imencode('.jpg', annotated)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        fps = 1.0 / (time.time() - start_time)
        print(f"🕒 FPS: {fps:.1f} | Detections: {len(results[0].boxes)} | Final Conclusion: {conclusion}")

        return jsonify({
            'image': f'data:image/jpeg;base64,{jpg_as_text}',
            'conclusion': conclusion
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🌐 Running Tomato Detection on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
