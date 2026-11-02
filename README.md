# Flood Detection: Production Deep Learning Pipeline

This workspace contains the complete production-grade training pipeline and assets for flood image classification with domain generalization.

---

## 📁 Workspace Directory Structure

```
Flood_Detection/
├── kaggle_flood_detection_pipeline.ipynb   <-- Production Kaggle Notebook
├── data/
│   └── AlleyFloodNet/
│       ├── flood_train_/
│       │   ├── flooded/          (468 images)
│       │   └── non_flooded/      (380 images)
│       └── flood_test_/          (Holdout test benchmark - 262 images)
│           ├── flooded/          (133 images)
│           └── non_flooded/      (129 images)
├── weights/
│   └── mobilenet_1_0_224_tf_no_top.h5    (Offline ImageNet pretrained weights)
└── README.md
```

---

## 🚀 How to Run on Kaggle (Step-by-Step)

### Step 1: Open Kaggle & Create a New Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. Click **Create** $\rightarrow$ **New Notebook**.

### Step 2: Upload Your Notebook
1. In the notebook editor, click **File** (top left menu) $\rightarrow$ **Import Notebook**.
2. Select `kaggle_flood_detection_pipeline.ipynb` from this folder (`Downloads/Flood_Detection/`).

### Step 3: Attach Datasets
Click **+ Add Input** (top right of Kaggle notebook) and attach:
1. **Flood Classification Dataset** (by `dhawalsrivastava2583`).
2. **AlleyFloodNet** (by `seonyseony`).
3. *(Optional if Internet is off)*: **mobilenet-v1-imagenet-weights** (by `aditisharma2007`).

### Step 4: Configure Hardware Accelerator
On the right-hand settings panel:
* **Accelerator:** Choose **GPU T4 x2** (or GPU P100).
* **Internet:** Toggle **ON** (allows automatic weight downloads).
* **Language:** Python 3.

### Step 5: Run the Notebook
* Click **Run All** (or run cell-by-cell).
* Estimated runtime on Tesla T4: **35–40 minutes**.

---

## 📊 Pipeline Architecture & Key Highlights

1. **Leakage-Safe Partitioning:**
   - Main dataset: 80% train (10,435 images) / 20% validation (2,609 images) with stratified sampling.
   - AlleyFloodNet test set (262 images) is strictly held out.
2. **Dynamic Class Weighting:**
   - Automatically balances the 2.48 : 1 ratio between flood and non-flood images.
3. **Data Augmentation:**
   - Random horizontal flips, rotations ($\pm 10^\circ$), zoom, and contrast adjustments prevent the network from memorizing simple water textures.
4. **Domain Balancing:**
   - 50/50 balanced sampling between general flood imagery and alleyway floods during fine-tuning.
5. **Grad-CAM Explainability:**
   - Generates activation heatmaps on misclassified samples to verify whether the model is focusing on flood debris or mistakenly flagging natural water bodies.
6. **Holdout Evaluation:**
   - Full evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, FPR, and FNR.
