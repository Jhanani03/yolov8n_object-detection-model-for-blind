
# YOLOv8n Object Detection Model for the Visually Impaired 🚶‍♀️👁️

Achieved high detection accuracy — correctly identifying most obstacles in real-time with an average precision (**mAP**) of **66%** and smooth guidance at approximately **77 FPS (≈6 ms/frame)**.

---

## 🚀 Overview

This project implements and fine-tunes the **Ultralytics YOLOv8n (Nano)** model for efficient, real-time obstacle detection to assist visually impaired navigation.
It delivers an excellent balance between **speed and accuracy**, optimized for deployment on edge devices.

---

## 🧠 Model Details

* **Base Model:** `YOLOv8n`
* **Framework:** `Ultralytics`
* **Training Type:** Transfer learning on a custom obstacle dataset
* **Output:** Bounding boxes, class labels, and confidence scores

---

## 📁 Repository Contents

| File / Folder          | Description                         |
| ---------------------- | ----------------------------------- |
| `robust_final_last.pt` | Final trained YOLOv8n model weights |
| `train_yolov8n.py`     | Training code used for fine-tuning  |
| `README.md`            | Documentation and usage guide       |

---

## ☁️ Cloud Training Advantage

This training pipeline is fully **Google Colab–compatible**, enabling seamless execution across different Google accounts.
By leveraging **Google Drive–based checkpoints**, training sessions can be **paused, resumed, or transferred** effortlessly — ensuring:

* 🔁 Continuous training flexibility (even if GPU limits reset)
* ☁️ Persistent data and model storage in the cloud
* 🤝 Easy collaboration and cross-device accessibility

This design makes the workflow **robust, portable, and ideal for research or real-world deployment** scenarios.

---




