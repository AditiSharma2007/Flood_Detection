import os
import sys
import json
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models

# Paths
BASE_DIR = r"c:\Users\Aditi Sharma\Downloads\Flood_Detection"
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
MODEL_PATH = os.path.join(MODELS_DIR, "flood_mobilenet_best.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Plot Confusion Matrix
# ==============================================================================
print("Generating Confusion Matrix visualization...")
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

cm = np.array(metrics["confusion_matrix"])
# cm: [[TP, FN], [FP, TN]]
# Row 0: True Floods (103 TP, 30 FN)
# Row 1: True Normal (8 FP, 121 TN)

fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

classes = ["Flooding", "Normal"]
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    title=f"AlleyFloodNet Confusion Matrix\n(Accuracy: {metrics['test_accuracy']*100:.2f}% | Precision: {metrics['flooding_precision']*100:.2f}%)",
    ylabel='Actual Label',
    xlabel='Predicted Label'
)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        label_text = f"{val}\n"
        if i == 0 and j == 0:
            label_text += "(True Flood)"
        elif i == 0 and j == 1:
            label_text += "(Missed)"
        elif i == 1 and j == 0:
            label_text += "(False Alarm)"
        else:
            label_text += "(True Normal)"

        ax.text(j, i, label_text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontweight='bold', fontsize=10)

fig.tight_layout()
cm_out = os.path.join(MODELS_DIR, "confusion_matrix.png")
plt.savefig(cm_out, bbox_inches='tight')
plt.close()
print(f"[OK] Saved Confusion Matrix to: {cm_out}")

# ==============================================================================
# 2. Performance Comparison Chart (Old vs New)
# ==============================================================================
print("Generating Performance Comparison Chart...")
categories = ["Test Accuracy", "Flood Precision", "Normal Recall", "FPR (False Alarms)"]
old_vals = [85.50, 86.00, 85.30, 14.70]
new_vals = [85.50, 92.79, 93.80, 6.20]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
rects1 = ax.bar(x - width/2, old_vals, width, label='Previous Model (MobileNetV1)', color='#9E9E9E')
rects2 = ax.bar(x + width/2, new_vals, width, label='New Model (MobileNetV2 + Augmentation)', color='#1976D2')

ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax.set_title('Model Performance Comparison on AlleyFloodNet Holdout Test Set', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 110)

def autolabel(rects, is_lower_better=False):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
comp_out = os.path.join(MODELS_DIR, "model_comparison.png")
plt.savefig(comp_out, bbox_inches='tight')
plt.close()
print(f"[OK] Saved Comparison Chart to: {comp_out}")

# ==============================================================================
# 3. Grad-CAM Explainability Heatmaps
# ==============================================================================
print("Generating Grad-CAM Explainability Heatmaps...")
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(64, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Hook into the last feature layer of MobileNetV2
gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

def compute_gradcam(img_path):
    gradients.clear()
    activations.clear()

    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        tensor = eval_transform(img_rgb).unsqueeze(0).to(device)

    model.zero_grad()
    output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]
    pred_idx = torch.argmax(probs).item()

    # Target: Class 0 (Flooding) score
    score = output[0, 0]
    score.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)

    # Resize CAM to image size
    img_pil = img_rgb.resize((224, 224))
    cam_img = Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)
    cam_np = np.array(cam_img) / 255.0

    return np.array(img_pil), cam_np, probs.cpu().detach().numpy()

# Test on 4 representative samples: 2 normal alley, 2 flooded alley
sample_normal_dir = r"D:\Flood_Detection_Data\AlleyFloodNet\flood_test_\non_flooded"
sample_flood_dir = r"D:\Flood_Detection_Data\AlleyFloodNet\flood_test_\flooded"

norm_files = [os.path.join(sample_normal_dir, f) for f in os.listdir(sample_normal_dir)[:2]]
flood_files = [os.path.join(sample_flood_dir, f) for f in os.listdir(sample_flood_dir)[:2]]
all_samples = [(f, "Normal") for f in norm_files] + [(f, "Flooding") for f in flood_files]

fig, axes = plt.subplots(4, 3, figsize=(12, 14), dpi=150)

for row_idx, (s_path, true_label) in enumerate(all_samples):
    img_np, cam, probs = compute_gradcam(s_path)
    pred_label = "Flooding" if probs[0] > 0.50 else "Normal"
    conf = probs[0]*100 if pred_label == "Flooding" else probs[1]*100

    # 1. Original Image
    axes[row_idx, 0].imshow(img_np)
    axes[row_idx, 0].set_title(f"True: {true_label}", fontsize=10, fontweight='bold')
    axes[row_idx, 0].axis('off')

    # 2. Grad-CAM Heatmap
    im_h = axes[row_idx, 1].imshow(cam, cmap='jet')
    axes[row_idx, 1].set_title("Flood Activation Map", fontsize=10)
    axes[row_idx, 1].axis('off')

    # 3. Superimposed Overlay
    heatmap_color = plt.cm.jet(cam)[:, :, :3]
    overlay = np.uint8((0.65 * (img_np / 255.0) + 0.35 * heatmap_color) * 255)
    axes[row_idx, 2].imshow(overlay)
    res_color = 'green' if pred_label == true_label else 'red'
    axes[row_idx, 2].set_title(f"Pred: {pred_label} ({conf:.1f}%)", fontsize=10, fontweight='bold', color=res_color)
    axes[row_idx, 2].axis('off')

plt.suptitle("Grad-CAM Explainability: Model Decision Heatmaps on Test Images", fontsize=13, fontweight='bold', y=0.99)
fig.tight_layout()
gradcam_out = os.path.join(MODELS_DIR, "gradcam_analysis.png")
plt.savefig(gradcam_out, bbox_inches='tight')
plt.close()
print(f"[OK] Saved Grad-CAM Visualizations to: {gradcam_out}")
print("\nALL REPORT VISUALIZATIONS GENERATED SUCCESSFULLY!")
