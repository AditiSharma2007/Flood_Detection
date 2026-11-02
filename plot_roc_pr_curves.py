"""
Generate ROC and Precision-Recall curves on the AlleyFloodNet test set.
Uses the saved best model checkpoint (flood_mobilenet_best.pt).
"""

import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_FLOOD  = r"c:\Users\Aditi Sharma\Downloads\Flood_Detection\data\AlleyFloodNet\flood_test_\flooded"
TEST_NORMAL = r"c:\Users\Aditi Sharma\Downloads\Flood_Detection\data\AlleyFloodNet\flood_test_\non_flooded"
MODEL_PATH  = r"c:\Users\Aditi Sharma\Downloads\Flood_Detection\models\flood_mobilenet_best.pt"
OUT_IMAGE   = r"c:\Users\Aditi Sharma\Downloads\Flood_Detection\models\roc_pr_curves.png"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Dataset ────────────────────────────────────────────────────────────────────
def get_images(folder):
    valid = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(valid) and os.path.getsize(os.path.join(folder, f)) > 0
    ]

class FloodDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.paths[idx]) as img:
                img = img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

flood_imgs  = get_images(TEST_FLOOD)
normal_imgs = get_images(TEST_NORMAL)
paths  = flood_imgs + normal_imgs
labels = [0] * len(flood_imgs) + [1] * len(normal_imgs)
print(f"Test set: {len(flood_imgs)} flooded, {len(normal_imgs)} non-flooded")

loader = DataLoader(
    FloodDataset(paths, labels, eval_transform),
    batch_size=32, shuffle=False, num_workers=0
)

# ── Model ──────────────────────────────────────────────────────────────────────
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
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# ── Inference ──────────────────────────────────────────────────────────────────
all_targets, all_probs = [], []
with torch.no_grad():
    for imgs, targets in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        # prob of class 0 = "flooded" is the positive class
        all_probs.extend(probs[:, 0].cpu().numpy())
        all_targets.extend(targets.numpy())

y_true  = np.array(all_targets)
y_score = np.array(all_probs)

# Flip labels for sklearn: 1 = flooded (positive), 0 = normal
y_true_bin = (y_true == 0).astype(int)

# ── Metrics ────────────────────────────────────────────────────────────────────
fpr_arr, tpr_arr, _ = roc_curve(y_true_bin, y_score)
roc_auc_val         = auc(fpr_arr, tpr_arr)

prec_arr, rec_arr, _ = precision_recall_curve(y_true_bin, y_score)
pr_auc_val            = average_precision_score(y_true_bin, y_score)

print(f"ROC-AUC : {roc_auc_val:.4f}")
print(f"PR-AUC  : {pr_auc_val:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor("#0F1117")

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32,
                       left=0.07, right=0.97, top=0.88, bottom=0.12)

ACCENT_ROC = "#00C8FF"
ACCENT_PR  = "#FF6B6B"
DIAG_CLR   = "#555566"
GRID_CLR   = "#2A2A3A"
TEXT_CLR   = "#E8E8F0"
FILL_ALPHA = 0.12

# ─── ROC Curve ────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#0F1117")
ax1.plot(fpr_arr, tpr_arr, color=ACCENT_ROC, lw=2.5,
         label=f"MobileNetV2  (AUC = {roc_auc_val:.4f})")
ax1.fill_between(fpr_arr, tpr_arr, alpha=FILL_ALPHA, color=ACCENT_ROC)
ax1.plot([0, 1], [0, 1], color=DIAG_CLR, lw=1.2, linestyle="--", label="Random Classifier")

# Annotate operating point (threshold ~0.5)
from sklearn.metrics import confusion_matrix
preds = (y_score >= 0.5).astype(int)
cm = confusion_matrix(y_true_bin, preds)
tp, fp = cm[1, 1], cm[0, 1]
fn, tn = cm[1, 0], cm[0, 0]
op_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
op_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
ax1.scatter([op_fpr], [op_tpr], color="#FFD700", s=90, zorder=5,
            label=f"Operating Point (thr=0.5)")
ax1.annotate(f"  ({op_fpr:.2f}, {op_tpr:.2f})",
             xy=(op_fpr, op_tpr), color="#FFD700",
             fontsize=8.5, va="bottom")

ax1.set_xlim([-0.01, 1.01])
ax1.set_ylim([-0.01, 1.05])
ax1.set_xlabel("False Positive Rate", color=TEXT_CLR, fontsize=11)
ax1.set_ylabel("True Positive Rate", color=TEXT_CLR, fontsize=11)
ax1.set_title("ROC Curve — AlleyFloodNet Test Set", color=TEXT_CLR, fontsize=13, fontweight="bold", pad=10)
ax1.tick_params(colors=TEXT_CLR)
ax1.spines[:].set_color(GRID_CLR)
ax1.grid(True, color=GRID_CLR, linewidth=0.7, linestyle="--")
legend1 = ax1.legend(loc="lower right", fontsize=9,
                     facecolor="#1A1A2E", edgecolor=GRID_CLR,
                     labelcolor=TEXT_CLR)

# ─── Precision-Recall Curve ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#0F1117")
ax2.plot(rec_arr, prec_arr, color=ACCENT_PR, lw=2.5,
         label=f"MobileNetV2  (AP = {pr_auc_val:.4f})")
ax2.fill_between(rec_arr, prec_arr, alpha=FILL_ALPHA, color=ACCENT_PR)

# Baseline (fraction of positives)
baseline = y_true_bin.mean()
ax2.axhline(y=baseline, color=DIAG_CLR, lw=1.2, linestyle="--",
            label=f"Baseline (class ratio = {baseline:.2f})")

# Operating point
from sklearn.metrics import precision_score, recall_score
op_prec = precision_score(y_true_bin, preds)
op_rec  = recall_score(y_true_bin, preds)
ax2.scatter([op_rec], [op_prec], color="#FFD700", s=90, zorder=5,
            label=f"Operating Point (thr=0.5)")
ax2.annotate(f"  ({op_rec:.2f}, {op_prec:.2f})",
             xy=(op_rec, op_prec), color="#FFD700",
             fontsize=8.5, va="top")

ax2.set_xlim([-0.01, 1.01])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("Recall", color=TEXT_CLR, fontsize=11)
ax2.set_ylabel("Precision", color=TEXT_CLR, fontsize=11)
ax2.set_title("Precision-Recall Curve — AlleyFloodNet Test Set", color=TEXT_CLR, fontsize=13, fontweight="bold", pad=10)
ax2.tick_params(colors=TEXT_CLR)
ax2.spines[:].set_color(GRID_CLR)
ax2.grid(True, color=GRID_CLR, linewidth=0.7, linestyle="--")
legend2 = ax2.legend(loc="lower left", fontsize=9,
                     facecolor="#1A1A2E", edgecolor=GRID_CLR,
                     labelcolor=TEXT_CLR)

# ─── Super-title ──────────────────────────────────────────────────────────────
fig.suptitle(
    "MobileNetV2 Flood Detector — AlleyFloodNet Test Set  |  262 Images",
    color=TEXT_CLR, fontsize=14, fontweight="bold", y=0.97
)

plt.savefig(OUT_IMAGE, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n[OK] Saved: {OUT_IMAGE}")

