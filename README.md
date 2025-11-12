# YOLOv8n Object Detection Model for the Visually Impaired 🚶‍♀️👁️

Achieved high detection accuracy — correctly identifying most obstacles in real-time with an average precision (mAP) of **66%** and smooth guidance at approximately **77 FPS (≈6 ms/frame)**.

---

## 🚀 Overview  
This project implements and fine-tunes the **Ultralytics YOLOv8n (Nano)** model for efficient, real-time obstacle detection to assist visually impaired navigation.  
It achieves an excellent trade-off between **speed and accuracy**, optimized for deployment on edge devices.

---

## 🧠 Model  
- **Base Model:** `YOLOv8n`  
- **Framework:** `Ultralytics`  
- **Training Type:** Transfer learning on a custom obstacle dataset  
- **Output:** Bounding boxes, class labels, and confidence scores  

The trained model file:  


The trained model is saved as:  
robust_final_last.pt
