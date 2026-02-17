---
title: Tomato Quality AI
emoji: 🍅
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# TomatoQualityAI 

**A YOLO-based machine learning system for automatic grading and sorting of tomatoes — built on a Raspberry Pi with camera sensors and deployed as a web app for easy access.**

>  **[Live Demo → tomato-grade-ai.vercel.app](https://tomato-grade-ai.vercel.app/)**

---

##  Achievements

This project was developed in a **24-hour hackathon** organized by **IIHR (Indian Institute of Horticultural Research)** under the **Horticulture Department of India, Bengaluru**.
We ranked among the **Top 5 teams** in the competition! 🌟

---

##  About the Project

Our goal is to assist **farmers and food processing units** by automating tomato quality detection, reducing human error, and speeding up the sorting process. The system classifies tomatoes into three grades:

| Grade | Description |
|-------|-------------|
| 🍅 **Ripe** | Ready for sale and consumption |
| 🍏 **Unripe** | Needs more time to mature |
| 🍂 **Damaged** | Bruised, rotten, or defective |

---

##  Hardware Origin — Raspberry Pi Prototype

> **This project was originally built and deployed on physical hardware, not just software.**

### The Physical Setup
We built a **real-time tomato sorting prototype** using:

- **Raspberry Pi 4** — the brain of the system, running the YOLO model for inference
- **Pi Camera Module / USB Webcam** — captures live video feed of tomatoes on a conveyor belt
- **Sensors** — proximity/IR sensors to detect when a tomato enters the scanning zone
- **Servo Motors** — controlled by GPIO pins to physically sort tomatoes into "Ripe", "Unripe", and "Damaged" bins based on the model's classification

### How It Worked

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Tomato on   │────▶│  IR Sensor   │────▶│  Pi Camera   │────▶│  YOLOv8 on   │
│  Conveyor    │     │  Triggers    │     │  Captures    │     │  Raspberry   │
│  Belt        │     │  Detection   │     │  Frame       │     │  Pi (CPU)    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  Classifies  │
                                                               │  Ripe/Unripe │
                                                               │  /Damaged    │
                                                               └──────┬───────┘
                                                                      │
                     ┌──────────────┐     ┌──────────────┐           │
                     │  Servo Motor │◀────│  GPIO Signal │◀──────────┘
                     │  Sorts into  │     │  Activates   │
                     │  Correct Bin │     │  Sorting     │
                     └──────────────┘     └──────────────┘
```

### From Hardware to Web 

For **easier access, demonstration, and evaluation purposes**, we evolved the project into a web application. The core YOLOv8 model remains the same — only the input/output layer changed from hardware (camera + servos) to software (browser + API).

>  **[Try the Web App → tomato-grade-ai.vercel.app](https://tomato-grade-ai.vercel.app/)**

---

##  How YOLOv8 Works — Architecture Deep Dive

**YOLO** (You Only Look Once) is a state-of-the-art, real-time object detection algorithm. Unlike traditional methods that scan an image multiple times, YOLO processes the **entire image in a single forward pass** through the neural network — making it extremely fast.

### YOLOv8 Architecture

```
                          YOLOv8 Architecture
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐               │
│  │ INPUT     │    │ BACKBONE  │    │ NECK      │               │
│  │           │    │ (CSPDark- │    │ (PANet +  │               │
│  │ 640×640   │───▶│  net)     │───▶│  FPN)     │               │
│  │ Image     │    │           │    │           │               │
│  └───────────┘    └───────────┘    └─────┬─────┘               │
│                                          │                      │
│                    ┌─────────────────────┬┴──────────────┐      │
│                    │                     │               │      │
│                    ▼                     ▼               ▼      │
│              ┌──────────┐         ┌──────────┐    ┌──────────┐ │
│              │ HEAD     │         │ HEAD     │    │ HEAD     │ │
│              │ (Large   │         │ (Medium  │    │ (Small   │ │
│              │  Objects)│         │  Objects)│    │  Objects)│ │
│              │ 80×80    │         │ 40×40    │    │ 20×20    │ │
│              └────┬─────┘         └────┬─────┘    └────┬─────┘ │
│                   │                    │               │       │
│                   ▼                    ▼               ▼       │
│              ┌─────────────────────────────────────────────┐   │
│              │          PREDICTION OUTPUT                   │   │
│              │  • Bounding Box Coordinates (x, y, w, h)   │   │
│              │  • Class Probabilities (Ripe/Unripe/Damaged)│   │
│              │  • Confidence Score                         │   │
│              └─────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components Explained

| Component | Role |
|-----------|------|
| **Backbone (CSPDarknet)** | Extracts visual features from the image at multiple scales using convolutional layers |
| **Neck (PANet + FPN)** | Combines features from different layers to detect objects of various sizes — small, medium, and large |
| **Detection Head** | Produces the final predictions — bounding boxes, class labels (Ripe/Unripe/Damaged), and confidence scores |
| **Anchor-Free Design** | YOLOv8 eliminates anchor boxes, directly predicting object centers for faster and cleaner results |
| **Decoupled Head** | Separates classification and localization into independent branches for better accuracy |

### Why YOLO for This Project?

-  **Speed** — Processes frames in real-time, even on a Raspberry Pi CPU
-  **Accuracy** — Single-pass detection means fewer errors than two-stage detectors
-  **Lightweight** — YOLOv8n (nano) variant runs efficiently on edge devices
-  **Easy to Fine-Tune** — Transfer learning with custom tomato dataset took only 80 epochs

---

##  How We Built It

### 1. Data Collection & Preprocessing
- **Sources:** Collected multiple datasets from Kaggle and open-source repositories
- **Cleaning:** All images were manually cleaned and reclassified into **Ripe**, **Unripe**, and **Damaged**
- **Augmentation:** Applied rotation, flipping, brightness adjustments, and mosaic augmentation to increase robustness
- **Removed** duplicate and low-quality images
- **Split:** Final dataset divided into **Training (80%)**, **Validation (10%)**, and **Testing (10%)**

### 2. Model Training
- **Base Training:** 80 epochs on the cleaned dataset using YOLOv8
- **Fine-Tuning:** Further trained with custom data collected at **IIHR** to improve accuracy on Indian tomato varieties
- **Backend:** PyTorch for deep learning, OpenCV for image handling, Ultralytics YOLOv8 framework for training and inference

### 3. Hardware Integration (Raspberry Pi)
- Deployed the trained `.pt` model on a **Raspberry Pi 4**
- Connected **camera sensors** for live video feed
- Used **IR/proximity sensors** to detect incoming tomatoes
- **Servo motors** controlled via GPIO to physically sort tomatoes into bins

### 4. Web App (For Demonstration)
- Wrapped the same model in a **Gradio** interface for browser-based access
- Deployed on **Hugging Face Spaces** as the backend API
- Built a custom **frontend** (HTML/CSS/JS) deployed on **Vercel** for a polished UI

---

##  Key Features

-  **Three-Class Detection:** Ripe, Unripe, and Damaged
-  **Real-Time Inference:** Works with live camera input
-  **Hardware Ready:** Designed to integrate with sorting machinery, conveyor belts, and robotic arms
-  **Web Demo Available:** Try it instantly at [tomato-grade-ai.vercel.app](https://tomato-grade-ai.vercel.app/)
-  **Scalable:** Easily adaptable for other fruits and produce

---

##  Tech Stack

| Category | Technology |
|----------|------------|
| **ML Model** | YOLOv8 (Ultralytics) |
| **Deep Learning** | PyTorch |
| **Image Processing** | OpenCV |
| **Data Analysis** | NumPy, Pandas |
| **Hardware** | Raspberry Pi 4, Pi Camera, Servo Motors, IR Sensors |
| **Web Backend** | Gradio, Hugging Face Spaces |
| **Web Frontend** | HTML, CSS, JavaScript |
| **Hosting** | Vercel (Frontend), Hugging Face (API) |
| **Language** | Python |

---

##  Live Demo

| Platform | Link |
|----------|------|
|  **Web App (Frontend)** | [tomato-grade-ai.vercel.app](https://tomato-grade-ai.vercel.app/) |
|  **Hugging Face Space (API)** | [swayamshetkar/Tomato_Quality_Detector](https://huggingface.co/spaces/swayamshetkar/Tomato_Quality_Detector) |

---

##  Installation & Local Usage

```bash
# Clone the repository
git clone https://github.com/swayamshetkar/Ai-Powered-Grading-And-sorting-of-tomatoes.git
cd Ai-Powered-Grading-And-sorting-of-tomatoes

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

---

##  Contributors

- **[@swayamshetkar](https://github.com/swayamshetkar)**

---
