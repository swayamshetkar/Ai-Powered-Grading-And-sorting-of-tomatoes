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

# TomatoQualityAI 🍅

**A YOLO-based machine learning project for automatic grading and sorting of tomatoes.**

## 🏆 Achievements
This project was developed in a **24-hour hackathon** organized by **IIHR (Indian Institute of Horticultural Research)** under the Horticulture Department of India.
We ranked among the **Top 5 teams** in the competition! 🌟

## 📖 About the Project
Our goal is to assist farmers and food processing units by automating tomato quality detection, reducing human error, and speeding up the sorting process.

### From Hardware to Web 🛠️ ➡️ 🌐
Originally, this project was deployed on a **Raspberry Pi** with camera sensors for a physical sorting mechanism. For easier access and demonstration purposes, we have evolved it into this **Web Application** to visualize the model's output and performance in real-time.

## 🚀 Key Features
- **Real-time Detection:** Instantly classifies tomatoes into three categories:
  - **Ripe** 🍅
  - **Unripe** 🍏
  - **Damaged** 🍂
- **High Efficiency:** Powered by YOLOv8 for fast and accurate inference.
- **Scalable:** Can be integrated with sorting machinery or robotic arms, and adapted for other fruits.

## 🏗️ How We Built It

### 1. Data Collection & Preprocessing
- **Sources:** Collected multiple datasets from Kaggle and open-source repositories.
- **Cleaning:** All images were cleaned and reclassified into Ripe, Unripe, and Damaged.
- **Augmentation:** Applied rotation, flipping, and brightness adjustments to increase robustness.
- **Split:** Final dataset divided into Training (80%), Validation (10%), and Testing (10%).

### 2. Model Training
- **Algorithm:** **YOLOv8** (You Only Look Once, Version 8) - state-of-the-art for real-time object detection.
- **Process:**
  1. Base training for **80 epochs** on the cleaned dataset.
  2. **Fine-tuned** with custom data collected by IIHR to ensure accuracy for Indian tomato varieties.
- **Tools:** PyTorch backend, OpenCV for image handling, and Ultralytics YOLOv8 framework.

## 💻 Tech Stack
- **YOLOv8:** Object detection & classification
- **Python:** Main development language
- **OpenCV:** Image processing
- **PyTorch:** Deep learning framework
- **NumPy / Pandas:** Data cleaning and analysis
- **Gradio:** Web Interface

## 📦 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/swayamshetkar/TomatoQualityAI.git
   cd TomatoQualityAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```

## 👥 Contributors
- **[@swayamshetkar](https://github.com/swayamshetkar)**
- **[@codeyatri-dev](https://github.com/codeyatri-dev)**

---
*Built with ❤️ for agriculture and innovation.*