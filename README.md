# Flood Detection: Production Deep Learning Pipeline

This repository contains the complete production-grade PyTorch deep learning pipeline for flood classification across both ground-level and aerial/drone perspectives.

---

## 📁 Workspace Directory Structure

```
flood_detection/
├── models/
│   ├── flood_mobilenet_best.pt           <-- Single production model checkpoint (PyTorch state_dict, 9.47 MB)
│   ├── evaluation_metrics.json           <-- Holdout benchmark metrics
│   ├── improved_benchmark_comparison.json<-- Comprehensive multi-dataset evaluation metrics
│   ├── confusion_matrix.png              <-- Confusion Matrix plot
│   └── gradcam_analysis.png              <-- Grad-CAM explainability heatmaps
├── data/                                 <-- Training & evaluation datasets (in .gitignore)
├── predict.py                            <-- 1-Click prediction script for single images
├── test_model_on_floodnet.py             <-- FloodNet UAV evaluation benchmark
├── test_model_on_zip.py                  <-- External test benchmark
├── gradcam.py                            <-- Grad-CAM explainability visualization
├── FINAL_PROJECT_REPORT.md               <-- Full benchmark performance report
├── requirements.txt                      <-- Python dependencies
└── README.md
```

---

## ⚡ Quick Start: Running Predictions

To run flood prediction on any image (ground or aerial perspective):

```powershell
.\.venv\Scripts\python.exe predict.py "path\to\image.jpg"
```

The script uses multi-scale ensemble inference and outputs:
* Classification: `FLOODING DETECTED` or `NORMAL / NON-FLOOD`
* Class Confidence: exact percentages for Flooding vs. Normal

---

## 📊 Model Performance Highlights

* **Model Architecture:** MobileNetV2 with custom dual-dropout classification head
* **Ground-Level Accuracy (AlleyFloodNet Test):** **92.37%** (ROC-AUC: 0.9851)
* **Aerial / Drone Accuracy (FloodNet UAV Test):** **83.48%** (ROC-AUC: 0.9214)
* **Natural Water False-Alarm Mitigation:** 100% rejection on swimming pools & calm water bodies
