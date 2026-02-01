# Road Surface Defect Detection and Assessment System
![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)
![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C)
![YOLOv5](https://img.shields.io/badge/YOLO-v5.7.0-00FFFF)

## 📖 Introduction
This repository contains the implementation of my Undergraduate Thesis: **"Detection Method for Rural Road Surface Defects Based on Deep Learning"**.

With the rapid increase in transportation volume, rural roads suffer from frequent damage. Traditional manual inspection is labor-intensive due to the long distances of rural highways. Furthermore, **asphalt patches (repair marks)** are often misclassified as defects by conventional detection models.

This project proposes a robust automated inspection system that:
1.  **Eliminates False Positives**: Explicitly trains on asphalt patches to distinguish them from actual cracks/potholes.
2.  **Quantifies Severity**: Combines **YOLOv5** detection with **Inverse Perspective Mapping (IPM)** and **U-Net** segmentation to calculate the physical area of defects.

## 🚀 Key Features
* **Hybrid Architecture**: Integration of Object Detection (YOLOv5) and Semantic Segmentation (U-Net).
* **Perspective Correction**: Implements IPM to transform angled dashboard camera views into bird's-eye views for accurate area measurement.
* **Robust Dataset**: Custom dataset collected under real-world driving conditions (60km/h), specifically addressing the "fake defect" issue caused by road repairs.

## 🛠️ Pipeline Overview

The system processes video data in four stages:

1.  **Data Acquisition**: captured via trunk-mounted camera (Depression angle: 25°, Speed: ~60km/h, 60fps).
2.  **Detection (YOLOv5)**: Localizes defects and repair patches.
3.  **Correction (IPM)**: Transforms the detected ROI (Region of Interest) to a top-down view to remove perspective distortion.
4.  **Evaluation (U-Net)**: Segments the defect area and calculates severity based on pixel-to-physical metrics.


## 📊 Dataset & Experiments

### Data Collection
* **Source**: Actual rural highway footage.
* **Preprocessing**: Manual screening and Data Augmentation (Horizontal Flip).
* **Total Images**: 2,874 images (1,437 original + 1,437 augmented).
* **Classes**: 
    * `Defect` (Cracks, Potholes)
    * `Patch` (Asphalt repair marks - **Crucial for reducing false positives**)

### Model Performance (mAP)
We compared three baseline models. YOLOv5 achieved the best performance, especially when handling the complex "Defects + Patch" dataset.

| Model | mAP (Defects Only) | mAP (Defects + Patch Dataset) |
| :--- | :---: | :---: |
| SSD | 0.769 | 0.772 |
| Faster R-CNN | 0.818 | 0.831 |
| **YOLOv5 (Ours)** | **0.846** | **0.890** |

> *Note: By incorporating "Patch" data, the model's ability to distinguish between actual damage and repairs significantly improved.*

## 📂 Project Structure

```text
Road-Defect-Detection-Assessment/
├── data/
│   ├── hyps/hyp.road.yaml     # Custom hyperparameters for road textures
│   └── road_defect.yaml       # Dataset configuration (2 classes)
├── models/
│   ├── yolo.py                # YOLOv5 architecture
│   ├── unet_custom.py         # Custom U-Net for segmentation
│   └── baselines.py           # SSD & Faster R-CNN implementations
├── utils/
│   ├── ipm_transform.py       # Inverse Perspective Mapping logic
│   └── road_dataset.py        # Dataloader for training
├── runs/
│   ├── train/                 # Training logs and weights
│   └── evaluation/            # Output videos with severity assessment
├── evaluate_pipeline.py       # Main script: YOLO -> IPM -> UNet
├── train.py                   # Training script
└── requirements.txt           # Dependencies (PyTorch 1.13.1)


⚙️ Quick Start
1. Installation
The environment is pinned to versions used during the thesis (2023) for reproducibility.

Bash
pip install -r requirements.txt
2. Run Evaluation Pipeline
This script runs the full detection and assessment flow on a video file.

Bash
python evaluate_pipeline.py \
    --source data/images/test_video.mp4 \
    --weights-yolo weights/best_defect.pt \
    --weights-unet weights/unet_road.pth
3. Training (Reproduction)
To reproduce the training results using the custom hyperparameter config:

Bash
python train.py --img 640 --batch 16 --epochs 100 --data road_defect.yaml --hyp data/hyps/hyp.road.yaml --weights yolov5s.pt
📝 Acknowledgements
This project was part of my undergraduate research. Special thanks to the laboratory for providing the raw video data and initial segmentation masks for U-Net training.
