#  TomatoQualityAI

A YOLO-based machine learning project for automatic grading and sorting of tomatoes into three categories: **ripe**, **unripe**, and **damaged**.

---

##  About the Project
This project was developed in a **24-hour hackathon** organized by **IIHR (Indian Institute of Horticultural Research)** under the **Horticulture Department of India**.  
We ranked among the **Top 5 teams** in the competition.

Our goal was to assist farmers and food processing units by automating tomato quality detection, reducing human error, and speeding up the sorting process.

---

##  How We Built It

###  Data Collection & Preprocessing
- We collected multiple tomato image datasets from **Kaggle** and other open-source repositories.  
- All images were **cleaned and reclassified** into three main categories:
  -  **Ripe**
  -  **Unripe**
  -  **Damaged**
- Performed **data augmentation** (rotation, flipping, brightness adjustments) to increase model robustness.  
- Removed duplicate and low-quality images.  
- Final dataset was split into **training (80%)**, **validation (10%)**, and **testing (10%)**.

###  Model Training
- Trained using **YOLOv8 (You Only Look Once, Version 8)** — a state-of-the-art real-time object detection algorithm.
- YOLO divides an image into grids and predicts bounding boxes and class probabilities simultaneously, making it **extremely fast and efficient** for real-time applications.
- Our training process:
  - Base training for **80 epochs** on cleaned dataset.
  - Fine-tuned the model with our **custom IIHR-collected data** for better accuracy in Indian tomato varieties.
- Used **PyTorch backend**, **OpenCV** for image handling, and **Ultralytics YOLOv8** framework for training and inference.

---

##  Tech Stack
-  **YOLOv8** — object detection & classification  
-  **Python** — main development language  
-  **OpenCV** — image processing  
-  **PyTorch** — deep learning framework  
-  **NumPy / Pandas** — data cleaning and analysis  

---

##  Features
- Detects and classifies tomatoes as:
  -  **Ripe**
  -  **Unripe**
  -  **Damaged**
- Works in **real-time** with camera input.
- Easy to **integrate with sorting machinery** or robotic arms.
- Scalable to other fruits or quality categories.

---

##  Dataset
Custom-collected and processed dataset derived from multiple Kaggle sources.  
Each image is annotated using YOLO format (`.txt` labels) containing bounding box coordinates and class IDs.
cd TomatoQualityAI
pip install -r requirements.txt
