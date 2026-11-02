# Research Paper Blueprint & Academic Manuscript Guide
## Ground-Level Flood Detection via MobileNetV2 with Hard-Negative Water Mining and Domain Adaptation

**Author:** Aditi Sharma  
**Target Conferences/Journals:** IEEE Access, Springer Multimedia Tools and Applications, Elsevier Computers & Geosciences, or IEEE International Conference on Image Processing (ICIP)

---

## 1. Suggested Paper Titles
Choose one based on your preferred emphasis:
1. **Primary Recommendation:**  
   *Mitigating Spurious Correlations in Ground-Level Flood Detection: A Lightweight MobileNetV2 Architecture with Hard-Negative Water Mining and Domain Adaptation*
2. **Edge / Real-Time Focus:**  
   *EdgeFloodNet: Real-Time Ground-Image Flood Detection on Edge Devices with Zero-False-Alarm Natural Water Calibration*
3. **Computer Vision & Disaster Management Focus:**  
   *Beyond Water Presence: Resolving Shortcut Learning in Disaster Computer Vision via Inverted Residuals and Hard Negative Calibration*

---

## 2. Abstract (Structured Academic Format)

> **Background:** Urban flash floods represent a critical threat to human life and infrastructure. While satellite remote sensing provides macro-level flood maps, its efficacy is severely constrained by prolonged revisit intervals, cloud obstructions, and inadequate spatial resolution in dense urban alleys. Ground-level computer vision leveraging closed-circuit television (CCTV) and crowd-sourced smartphone imagery offers immediate, hyper-local situational awareness.  
> **Problem Statement:** Standard binary flood classification models suffer from severe *shortcut learning* (spurious correlation). Because training datasets typically contrast flooded scenes against completely dry roads and buildings, neural networks inadvertently learn that any visible water surface constitutes a flood—causing catastrophic false-positive alert rates (exceeding 98% confidence) when encountering calm natural rivers, lakes, scenic meadow streams, and beaches.  
> **Proposed Methodology:** In this paper, we propose an edge-deployable, transfer-learning framework based on MobileNetV2 featuring an inverted residual architecture with linear bottlenecks. To eliminate the water-presence shortcut, we introduce a targeted **Hard-Negative Mining Pipeline** incorporating 75+ diverse natural water bodies directly into the non-flood class under a dynamically weighted sampling regime ($20\times$ oversampling) with aggressive affine and chromatic augmentations. Furthermore, we implement domain adaptation on the challenging AlleyFloodNet benchmark to address confined urban alleyways.  
> **Results:** Evaluated on an independent, untouched test holdout of 262 real-world images, the proposed framework elevates overall classification accuracy from 85.50% to **90.46%**, achieving an **ROC-AUC of 0.9758** and **PR-AUC of 0.9802**. More importantly, false alarms on natural rivers and water bodies are reduced from 98.20% false flooding confidence down to **2.37%–5.87%**, while maintaining a high flood recall of **90.23%** (120/133 true floods detected). The model consumes only 9.47 MB of storage with sub-15ms inference latency, demonstrating outstanding viability for real-time edge deployment.

---

## 3. Introduction Section

### 3.1 Motivation
* Climate change has intensified the frequency of extreme precipitation events and urban flash floods.
* Satellite Synthetic Aperture Radar (SAR) and multispectral optical sensors suffer from:
  1. High latency (hours to days revisit intervals).
  2. Severe cloud/rain attenuation.
  3. "Urban canyon" occlusion where tall buildings obscure ground water levels.
* Ground-level visual sensors (smartphones, dashcams, municipal CCTVs) provide zero-latency, high-resolution visual evidence for first responders.

### 3.2 The Fundamental Research Problem: Shortcut Learning
* Deep Convolutional Neural Networks (CNNs) are notorious for exploiting high-frequency spurious shortcuts (Geirhos et al., 2020).
* In conventional datasets (e.g., Dhawal Srivastava Flood Dataset), Class 0 (Flooded) contains water, while Class 1 (Non-Flooded) contains dry pavement, dry vehicles, and sunlit structures.
* Consequently, the loss function minimizes empirical risk by detecting:
  $$\hat{y} = f(\text{Liquid Surface Reflectance}) \quad \text{instead of} \quad f(\text{Infrastructure Inundation})$$
* This causes unacceptable false alarm rates in scenic river valleys, parks, and beach environments.

### 3.3 Key Contributions
1. **Architectural Transition to MobileNetV2:** Replaced traditional heavy backbones with inverted residual blocks and linear bottlenecks, preventing non-linearities from destroying manifold representations in lower-dimensional latent spaces.
2. **Hard-Negative Natural Water Calibration:** Developed an automated hard-negative mining procedure collecting natural rivers, streams, lakes, and canals, oversampled at $20\times$ with stochastic color jittering to enforce structural discrimination over color/texture shortcuts.
3. **Domain Adaptation for Confined Urban Alleys:** Successfully transferred features across macro-disaster imagery and narrow urban alleyways using weighted cross-entropy and selective top-block unfreezing.
4. **Empirical Validation with Rigorous Holdout Benchmark:** Conducted extensive benchmark evaluations on an untouched 262-image AlleyFloodNet test set, achieving 90.46% accuracy, 0.9758 ROC-AUC, and 0.9802 PR-AUC, all without any test-set leakage.
5. **Grad-CAM Visual Explainability Module:** Implemented a full Gradient-weighted Class Activation Mapping (Grad-CAM) system on MobileNetV2's final convolutional block, generating pixel-level heatmaps that visually confirm the model attends to *submerged infrastructure* (submerged vehicles, flooded roads, water-logged walls) rather than general water surfaces. This provides both model transparency for real-world deployment and a compelling explainability figure for publication.

---

## 4. Methodology & Theoretical Formulation

### 4.1 Backbone: MobileNetV2 Inverted Residuals & Linear Bottlenecks
MobileNetV2 introduces the inverted residual block with depthwise separable convolutions:
1. **Expansion Layer:** A $1 \times 1$ pointwise convolution expands the low-dimensional input channel dimension $k$ to a higher dimensional space $tk$ (expansion ratio $t = 6$).
2. **Depthwise Convolution:** A $3 \times 3$ depthwise convolution applies spatial filtering with significantly reduced FLOPs:
   $$\text{FLOPs}_{\text{separable}} = h \cdot w \cdot d_{\text{in}} \cdot (k^2 + d_{\text{out}})$$
   yielding an $\approx 8\times$ computational efficiency compared to standard convolutions.
3. **Linear Bottleneck:** ReLU6 non-linearity is omitted in the projection layer to prevent destructive manifold collapse on zero-thresholding.

### 4.2 Mathematical Loss Formulation

#### 4.2.1 Dynamic Class-Weighted Cross-Entropy
Given class imbalance $N_{\text{flood}} : N_{\text{normal}} \approx 2.48 : 1$, we formulate weighted cross-entropy loss:
$$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^B \sum_{c=0}^1 w_c \cdot y_{i,c} \log(\hat{y}_{i,c})$$
where the dynamic class weights $w_c$ are defined inversely proportional to class frequencies:
$$w_c = \frac{N_{\text{total}}}{2 \cdot N_c}$$

#### 4.2.2 Hard Negative Sampling Distribution
To break shortcut learning, samples are drawn according to a categorical distribution parameterized by sample weights $\Omega$:
$$P(x_i) = \frac{\Omega_i}{\sum_{j=1}^N \Omega_j}$$
where:
$$\Omega_i = \begin{cases} 
20.0, & \text{if } x_i \in \mathcal{D}_{\text{hard\_negative\_water}} \\
4.0, & \text{if } x_i \in \mathcal{D}_{\text{AlleyFloodNet\_train}} \\
1.0, & \text{if } x_i \in \mathcal{D}_{\text{main\_dataset}}
\end{cases}$$

### 4.3 Grad-CAM Explainability Formulation
To verify that the model attends to submerged objects rather than general liquid surfaces:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
where $A^k$ represents the activation map of the final convolutional layer of MobileNetV2.

### 4.4 Grad-CAM Explainability — Mathematical Formulation

For model interpretability, Gradient-weighted Class Activation Mapping (Grad-CAM) computes the class-discriminative localization map $L^c_{\text{Grad-CAM}} \in \mathbb{R}^{u \times v}$ for class $c$:

**Step 1:** Compute gradient of class score $y^c$ w.r.t. final convolutional feature map $A^k$ (MobileNetV2 `features[-1]`, shape $1280 \times 7 \times 7$):
$$\alpha_k^c = \underbrace{\frac{1}{Z} \sum_{i} \sum_{j}}_{\text{Global Average Pool}} \frac{\partial y^c}{\partial A^k_{ij}}$$

**Step 2:** Form weighted combination with ReLU to focus only on positive evidence for class $c$:
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

**Step 3:** Bilinear upsampling from $7 \times 7$ to original image resolution for pixel-level overlay.

The resulting heatmaps confirm that for genuine floods, the model activates over *submerged vehicles, flooded roads, and water-logged walls*, while for natural rivers, activations shift to *architectural elements and river banks*.

---

## 5. Experimental Setup & Datasets

### 5.1 Dataset Demographics Table

| Dataset Component | Source | Flooded (Class 0) | Normal / Non-Flood (Class 1) | Purpose in Research |
| :--- | :--- | :--- | :--- | :--- |
| **Main Dataset** | Dhawal Srivastava (Kaggle) | 9,296 | 3,748 | General feature representation |
| **AlleyFloodNet Train** | Seony et al. | 468 | 380 | Domain adaptation to urban alleys |
| **Water Hard Negatives** | Wikimedia / Natural Corpus | 0 | 75 | Eliminates natural river/beach false alarms |
| **Untouched Test Benchmark** | AlleyFloodNet (Held-out) | 133 | 129 | Unbiased rigorous benchmark (262 images) |

### 5.2 Training Hyperparameters

| Hyperparameter | Phase 1 (Baseline) | Phase 2 (Domain Adaptation) | Phase 3 (Calibrated Fine-Tuning) |
| :--- | :--- | :--- | :--- |
| **Base Learning Rate** | $1 \times 10^{-3}$ | $5 \times 10^{-5}$ | $1 \times 10^{-4}$ (Classifier), $3 \times 10^{-5}$ (Top Features) |
| **Optimizer** | AdamW | AdamW | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay=$10^{-4}$) |
| **Batch Size** | 32 | 32 | 32 |
| **Input Resolution** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ |
| **Data Augmentation** | Resize only | Flips, $\pm 10^\circ$ Rot | Random Flips, $\pm 15^\circ$ Rot, ColorJitter($\pm 0.20$) |
| **Hardware** | NVIDIA RTX 2050 (4GB) | NVIDIA RTX 2050 (4GB) | NVIDIA RTX 2050 (4GB) |
| **Compute Time** | ~4.2 mins | ~3.8 mins | ~4.9 mins |

---

## 6. Empirical Results & Comparative Analysis

### 6.1 Benchmark Comparison on AlleyFloodNet Holdout (262 Images)

| Evaluation Metric | Historical Baseline (MobileNetV1) | Intermediate Phase (AlleyNet Adapted) | **Proposed Calibrated Framework (Ours)** | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Test Accuracy** | 85.50% | 85.50% | **90.46%** | **+4.96% absolute gain** |
| **Flooding Precision** | 86.00% | 92.79% | **90.91%** | High alert reliability |
| **Flooding Recall** | 85.70% | 77.44% | **90.23%** | **120 out of 133 floods detected** |
| **Flooding F1-Score** | 85.85% | 84.43% | **90.57%** | **+4.72% gain** |
| **ROC-AUC** | 0.8800 | 0.9395 | **0.9758** | **+0.0958 gain** |
| **PR-AUC** | — | 0.9481 | **0.9802** | Near-optimal precision-recall area |
| **False Positive Rate (FPR)**| 14.70% | 6.20% | **9.30%** | Robust noise rejection |

### 6.2 Confusion Matrix (Untouched Benchmark)
```
                  Predicted Flooding (0)    Predicted Normal (1)
Actual Flood (0)          120 (TP)                  13 (FN)
Actual Normal (1)          12 (FP)                 117 (TN)
```

### 6.3 Grad-CAM Qualitative Analysis

A crucial qualitative contribution of this work is the Grad-CAM visual comparison between genuine floods and hard-negative natural water bodies:

| Scene Type | Class Prediction | Grad-CAM Focus Region | Interpretation |
| :--- | :--- | :--- | :--- |
| **Urban flash flood** | 🚨 FLOOD (99.83%) | Cars, road surface, walls | Correctly attends to submerged infrastructure |
| **Vehicles in muddy water** | 🚨 FLOOD (100.0%) | Vehicle bodies, waterline | Correctly detects human-built objects flooded |
| **Scenic meadow river** | ✅ NORMAL (94.13%) | River bank edges, trees | Correctly ignores calm natural water |
| **Mountain forest stream** | ✅ NORMAL (97.63%) | Forest canopy, rocks | Structural context correctly rejected |

These heatmaps serve as the paper's **Figure 4: Grad-CAM Explainability Analysis**, stored at:
`models/gradcam_analysis.png` (300 DPI, publication-ready).

> **How to generate for your own images:**
> ```powershell
> D:\Flood_Detection_Data\venv\Scripts\python.exe gradcam.py "C:\path\to\your_image.jpg"
> ```
> Output is saved to `gradcam_outputs\<image_name>_gradcam.png` automatically.

### 6.4 Real-World Natural Water Hard-Negative Ablation
Direct experimental verification on challenging non-flood natural water landscapes:

| Image Description | Baseline Model Prediction | Calibrated Model Prediction | Outcome |
| :--- | :--- | :--- | :--- |
| **Scenic Meadow River with Tree** | 🚨 **FLOOD (98.20% conf)** | **✅ NORMAL (94.13% conf)** | **Shortcut Broken** |
| **Mountain Forest Stream & Rapids** | 🚨 **FLOOD (74.33% conf)** | **✅ NORMAL (97.63% conf)** | **Shortcut Broken** |
| **Calm Lake Reflection** | 🚨 **FLOOD (91.45% conf)** | **✅ NORMAL (96.50% conf)** | **Shortcut Broken** |
| **Urban Flash Flood (Submerged Street)**| 🚨 **FLOOD (99.83% conf)** | 🚨 **FLOOD (99.83% conf)** | **True Flood Retained** |
| **Vehicles Submerged in Muddy Water** | 🚨 **FLOOD (100.0% conf)** | 🚨 **FLOOD (100.0% conf)** | **True Flood Retained** |

---

## 7. Edge Deployment & Computational Complexity

* **Model File Size:** 9.47 MB (PyTorch state dict), 9.51 MB (Full traced TorchScript model).
* **Parameters:** 2,298,882 (2.30 M parameters total).
* **Inference Latency (NVIDIA GeForce RTX 2050):** 8.2 ms per frame.
* **Inference Latency (Intel CPU i5/i7):** 24.6 ms per frame (over 40 FPS, well above real-time 30 FPS video).
* **Memory Footprint in VRAM:** ~777 MiB during batch inference.

---

## 8. Discussion & Future Research Directions
1. **Temporal Video Smoothing:** Adding temporal consistency filters (exponential moving average over 15-frame rolling windows) to prevent single-frame flickering in continuous CCTV feeds.
2. **Multi-Task Water Depth Segmentation:** Extending from binary classification to semantic segmentation of water depth levels (e.g., passable for pedestrians, passable for SUVs, hazard level).
3. **Synthetic Generative Augmentation:** Utilizing generative diffusion models to synthesize flooded variants of dry street scenes while preserving background architecture.

---

## 9. Key Citations (BibTeX Ready)

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}

@article{geirhos2020shortcut,
  title={Shortcut learning in deep neural networks},
  author={Geirhos, Robert and Jacobsen, J{\"o}rn-Henrik and Michaelis, Claudio and Zemel, Richard and Brendel, Wieland and Bethge, Matthias and Wichmann, Felix A},
  journal={Nature Machine Intelligence},
  volume={2},
  number={11},
  pages={665--673},
  year={2020},
  publisher={Nature Publishing Group}
}

@inproceedings{selvaraju2017grad,
  title={Grad-cam: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedaldi, Andrea and Parikh, Devi and Batra, Dhruv},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={618--626},
  year={2017}
}

@inproceedings{shrivastava2016training,
  title={Training region-based object detectors with online hard example mining},
  author={Shrivastava, Abhinav and Gupta, Abhinav and Girshick, Ross},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={761--769},
  year={2016}
}
```
