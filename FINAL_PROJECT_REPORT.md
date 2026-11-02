# Ground-Image Flood Detection: MobileNet Transfer Learning, Domain Adaptation & False Alarm Mitigation

**Project By:** Aditi Sharma  
**Target:** Binary Classification (`Flooding` vs `Normal`)  
**Hardware & Environment:** NVIDIA GeForce RTX 2050 (4 GB GDDR6), Python 3.11, PyTorch 2.5.1 + CUDA 12.1  
**Datasets:** Flood Classification Dataset (13,044 images) + AlleyFloodNet (1,110 images)

---

## 1. Executive Summary & Problem Solved

### The Core Problem: False Alarms on Normal Water Bodies (Beaches & Oceans)
In traditional flood classification datasets, the **"Flood"** class consists exclusively of images with water, while the **"Non-Flood"** class consists solely of dry asphalt roads and dry buildings. Consequently, deep convolutional neural networks learn a **spurious shortcut**:
$$\text{Water Present} \implies \text{Class: Flooding}$$

When presented with non-flood scenes containing natural water (beaches, coastlines, calm rivers, swimming pools, wet pavement), the model triggers high-confidence flood alarms.

### The Solution Implemented:
1. **Backbone Upgrade:** Transitioned from MobileNetV1 to **MobileNetV2** (utilizing Inverted Residual Blocks and Linear Bottlenecks to better capture fine contextual details, such as flood debris vs. natural coastlines).
2. **Aggressive Data Augmentation:** Implemented stochastic horizontal flips, rotations ($\pm 10^\circ$), and color/contrast jitter during training, preventing the model from memorizing static surface textures.
3. **Dynamic Class Weighting:** Countered the 2.48:1 class imbalance using inversely proportional cross-entropy weights.
4. **Domain Adaptation with AlleyFloodNet:** Fine-tuned the top residual feature blocks with weighted sampling, adapting the model to narrow urban alleyways.
5. **Explainability with Grad-CAM:** Validated feature activations to confirm the model targets true structural flood markers (submerged cars, walls, debris) rather than general water bodies.

---

## 2. Empirical Benchmark Results (Untouched AlleyFloodNet Holdout Test Set)

The model was evaluated on the **262 independent holdout images** of AlleyFloodNet (133 flooded, 129 non-flooded):

| Performance Metric | Historical Model (MobileNetV1) | Phase 2 (AlleyNet) | **Phase 3 (Calibrated + Hard Negatives)** |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 85.50% | 85.50% | **90.46%** (Highest overall accuracy) |
| **Flooding Precision** | 86.00% | 92.79% | **90.91%** (Reliable alerting) |
| **Flooding Recall** | 85.70% | 77.44% | **90.23%** (120 of 133 real floods caught!) |
| **Flooding F1-Score** | 85.85% | 84.43% | **90.57%** |
| **ROC-AUC** | 0.8800 | 0.9395 | **0.9758** (Exceptional separation) |
| **PR-AUC** | — | 0.9481 | **0.9802** |
| **False Positive Rate (FPR)** | 14.70% | 6.20% | **9.30%** |
| **River/Beach False Alarm Fix** | ❌ 98.2% False Alert | ❌ 74.3% False Alert | **✅ 94.1% - 97.6% Correct Normal!** |

### Confusion Matrix Breakdown (262 AlleyFloodNet Test Holdout):
* **True Positives (Floods Correctly Detected):** 120 / 133 (90.23% recall)
* **True Negatives (Normal Correctly Detected):** 117 / 129 (90.70% recall)
* **False Positives (Normal Misclassified as Flood):** 12
* **False Negatives (Floods Missed):** 13

---

## 3. Visual Deliverables & Artifacts Generated

All deliverables are saved in the project directory:

```
Flood_Detection/
├── models/
│   ├── flood_detection_final.pt     <-- Final production model (9.51 MB)
│   ├── flood_mobilenet_best.pt      <-- Best checkpoint weights (9.47 MB)
│   ├── evaluation_metrics.json      <-- Machine-readable benchmark metrics
│   ├── confusion_matrix.png         <-- High-res Confusion Matrix plot
│   ├── model_comparison.png        <-- Visual comparison bar chart
│   └── gradcam_analysis.png         <-- Grad-CAM explainability heatmaps
├── predict.py                       <-- 1-Click test tool for new images
├── train_local.py                   <-- Complete reproducible training script
└── README.md
```

---

## 4. How to Test Any Image (Inference Guide)

To test any new photo (beach, river, street, flooded area):

```powershell
D:\Flood_Detection_Data\venv\Scripts\python.exe predict.py "C:\path\to\your_image.jpg"
```

The script will output:
* Predicted Class (`FLOODING DETECTED` or `NORMAL / NON-FLOOD`)
* Exact confidence percentage for both classes
