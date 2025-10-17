# save as label_browser.py
from flask import Flask, render_template_string, request, redirect, send_from_directory
import os
import json
from PIL import Image

IMAGE_FOLDER = "new"  # folder with images
LABEL_FOLDER = "texts"  # folder to save .txt
os.makedirs(LABEL_FOLDER, exist_ok=True)

app = Flask(__name__)

images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
images.sort()
current_index = 0  # track which image we're labeling

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>YOLO Labeler - Drag to Draw Box</title>
<style>
body { font-family: Arial; text-align: center; }
#container { position: relative; display: inline-block; }
#img { max-width: 800px; max-height: 600px; }
#canvas { position: absolute; top: 0; left: 0; cursor: crosshair; }
</style>
</head>
<body>
<h3>Labeling image {{ image_name }} ({{ index+1 }}/{{ total }})</h3>

<div id="container">
  <img id="img" src="{{ url }}" onload="initCanvas()">
  <canvas id="canvas"></canvas>
</div>

<form method="POST">
  <input type="hidden" id="data" name="data">
  <select name="class_id">
    <option value="0">Ripe</option>
    <option value="1">Unripe</option>
    <option value="2">Damaged</option>
  </select>
  <button type="submit">Save</button>
</form>

<p>➡ Click and drag to draw a bounding box.</p>

<script>
let canvas, ctx, img;
let startX, startY, isDrawing = false;

function initCanvas() {
  img = document.getElementById('img');
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');

  // Match canvas to displayed image
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + "px";
  canvas.style.height = img.clientHeight + "px";

  // Add mouse listeners
  canvas.addEventListener('mousedown', startDraw);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', endDraw);
}

function startDraw(e) {
  const rect = canvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
  isDrawing = true;
}

function draw(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "red";
  ctx.lineWidth = 2;
  ctx.strokeRect(startX, startY, x - startX, y - startY);
}

function endDraw(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;
  isDrawing = false;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;
  ctx.strokeRect(startX, startY, endX - startX, endY - startY);

  // Store coordinates relative to displayed image
  document.getElementById('data').value = JSON.stringify({
    start: [startX, startY],
    end: [endX, endY],
    displayWidth: img.clientWidth,
    displayHeight: img.clientHeight
  });
  alert("Box drawn! Now select class and press Save.");
}
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def label_page():
    global current_index
    if current_index >= len(images):
        return "<h2>✅ All images labeled!</h2>"

    image_name = images[current_index]
    img_path = os.path.join(IMAGE_FOLDER, image_name)
    w, h = Image.open(img_path).size

    if request.method == "POST":
        data = json.loads(request.form["data"])
        class_id = int(request.form["class_id"])

        start = data["start"]
        end = data["end"]
        dw = data["displayWidth"]
        dh = data["displayHeight"]

        # Scale from display size → actual image size
        scale_x = w / dw
        scale_y = h / dh
        x1, y1 = start[0] * scale_x, start[1] * scale_y
        x2, y2 = end[0] * scale_x, end[1] * scale_y

        # YOLO format (normalized)
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h

        label_path = os.path.join(LABEL_FOLDER, os.path.splitext(image_name)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        current_index += 1
        return redirect("/")

    url = f"/img/{image_name}"
    return render_template_string(HTML, url=url, image_name=image_name, index=current_index, total=len(images))

@app.route("/img/<filename>")
def send_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
