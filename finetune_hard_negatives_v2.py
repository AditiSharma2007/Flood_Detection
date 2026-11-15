"""
Hard-Negative Calibration v3 — CORRECT label convention
  Class 0 = Flooding (matches predict.py probs[0] and train_local.py)
  Class 1 = Normal

Fixes two failure modes:
  A) Natural rivers / calm water → wrongly predicted as Flood
  B) Completely unrelated images (code screenshots, UI) → wrongly predicted as Flood
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import shutil
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, "models", "flood_mobilenet_best.pt")
BACKUP_PATH = os.path.join(BASE, "models", "flood_mobilenet_backup.pt.bak")
ALLEY       = os.path.join(BASE, "data", "AlleyFloodNet")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CORRECT class convention (matches train_local.py and predict.py) ──────────
# Class 0 = Flooding,  Class 1 = Normal
FLOOD_LABEL  = 0
NORMAL_LABEL = 1

print(f"Device: {DEVICE}", flush=True)
print(f"Label convention: Flood={FLOOD_LABEL}, Normal={NORMAL_LABEL}", flush=True)

# ── Hard-negative images the user flagged as misclassified ────────────────────
# Both should be Normal (label 1)
PRIORITY_NEGATIVES = [
    (os.path.join(BASE, "test_images", "user_normal_river.jpg"),    NORMAL_LABEL),
    (os.path.join(BASE, "test_images", "user_code_screenshot.png"), NORMAL_LABEL),
]

# ── Collect AlleyFloodNet ─────────────────────────────────────────────────────
def collect_alley(split_dir):
    items = []
    # flooded → label 0 = FLOOD_LABEL
    for lbl, folder_name in [(FLOOD_LABEL, "flooded"), (NORMAL_LABEL, "non_flooded")]:
        folder = os.path.join(ALLEY, split_dir, folder_name)
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    items.append((os.path.join(folder, f), lbl))
    return items

alley_train = collect_alley("flood_train_")
alley_val   = collect_alley("flood_test_")  # holdout — NEVER used in training

n_flood_train  = sum(1 for _, l in alley_train if l == FLOOD_LABEL)
n_normal_train = sum(1 for _, l in alley_train if l == NORMAL_LABEL)
n_flood_val    = sum(1 for _, l in alley_val   if l == FLOOD_LABEL)
n_normal_val   = sum(1 for _, l in alley_val   if l == NORMAL_LABEL)
print(f"AlleyFloodNet train: {len(alley_train)} (Flood={n_flood_train}, Normal={n_normal_train})", flush=True)
print(f"AlleyFloodNet holdout val: {len(alley_val)} (Flood={n_flood_val}, Normal={n_normal_val})", flush=True)

# ── Verify priority negatives exist ──────────────────────────────────────────
valid_pn = []
for path, lbl in PRIORITY_NEGATIVES:
    if os.path.exists(path):
        valid_pn.append((path, lbl))
        print(f"  Priority negative (Normal): {os.path.basename(path)}", flush=True)
    else:
        print(f"  WARNING: not found: {path}", flush=True)

# ── Build training set: AlleyFloodNet + oversampled hard negatives ─────────────
OVERSAMPLE_FACTOR  = 10   # each priority negative appears 10x per epoch
priority_oversampled = valid_pn * OVERSAMPLE_FACTOR
all_train = alley_train + priority_oversampled

n_flood_tot  = sum(1 for _, l in all_train if l == FLOOD_LABEL)
n_normal_tot = sum(1 for _, l in all_train if l == NORMAL_LABEL)
print(f"\nTotal train: {len(all_train)} (Flood={n_flood_tot}, Normal={n_normal_tot})", flush=True)
print(f"  incl. {len(priority_oversampled)} hard-neg copies", flush=True)

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class FloodDS(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            with Image.open(path) as img:
                return self.tf(img.convert("RGB")), label
        except Exception:
            return torch.zeros(3, 224, 224), label

train_ds = FloodDS(all_train, train_tf)
val_ds   = FloodDS(alley_val, val_tf)

# Balanced sampler
labels_list  = [l for _, l in all_train]
class_counts = [labels_list.count(0), labels_list.count(1)]
weights      = [1.0 / class_counts[l] for l in labels_list]
sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,   num_workers=0)
print(f"Batches/epoch: {len(train_loader)}", flush=True)

# ── Load model ────────────────────────────────────────────────────────────────
def build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(1280, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2),
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE)

model = build_model()

# FREEZE ALL features — only train classifier
for p in model.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,} (classifier only)", flush=True)

shutil.copy2(MODEL_PATH, BACKUP_PATH)
print(f"Backup saved to {BACKUP_PATH}", flush=True)

# ── Validation helpers ────────────────────────────────────────────────────────
def validate(tag="val"):
    model.eval()
    correct = total = 0
    flood_correct = flood_total = 0
    normal_correct = normal_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += preds.eq(lbls).sum().item()
            total   += lbls.size(0)
            # Flood = label 0
            fm = lbls == FLOOD_LABEL
            flood_correct  += preds[fm].eq(FLOOD_LABEL).sum().item()
            flood_total    += fm.sum().item()
            # Normal = label 1
            nm = lbls == NORMAL_LABEL
            normal_correct += preds[nm].eq(NORMAL_LABEL).sum().item()
            normal_total   += nm.sum().item()
    acc    = 100 * correct / total
    f_rec  = 100 * flood_correct  / flood_total  if flood_total  > 0 else 0
    n_rec  = 100 * normal_correct / normal_total if normal_total > 0 else 0
    print(f"  [{tag}] Acc={acc:.2f}%  Flood-Recall={f_rec:.2f}%  Normal-Recall={n_rec:.2f}%  ({total} images)", flush=True)
    return acc, f_rec, n_rec


def test_priority_negatives():
    model.eval()
    for path, expected_lbl in valid_pn:
        with Image.open(path) as img:
            t = val_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1)[0]
        flood_p  = probs[FLOOD_LABEL].item()  * 100   # index 0 = Flood
        normal_p = probs[NORMAL_LABEL].item() * 100   # index 1 = Normal
        pred_lbl = FLOOD_LABEL if flood_p > 50 else NORMAL_LABEL
        icon = "✅ NORMAL" if pred_lbl == NORMAL_LABEL else "🚨 FLOOD"
        status = "CORRECT" if pred_lbl == expected_lbl else "WRONG"
        print(f"    {icon} | Flood={flood_p:.1f}% Normal={normal_p:.1f}% | {status} | {os.path.basename(path)}", flush=True)


# ── Baseline ──────────────────────────────────────────────────────────────────
print("\n── Baseline (before training) ──────────────────────────────────", flush=True)
base_acc, base_frec, base_nrec = validate("holdout")
print("  Priority negatives (target: NORMAL):", flush=True)
test_priority_negatives()

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-6, weight_decay=1e-3
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_acc = base_acc
EPOCHS   = 8

print(f"\n── Training ({EPOCHS} epochs, classifier only, lr=5e-6) ─────────────────", flush=True)
print("=" * 65, flush=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_correct = tr_total = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        tr_correct += out.detach().argmax(1).eq(lbls).sum().item()
        tr_total   += lbls.size(0)

    tr_acc = 100 * tr_correct / tr_total
    print(f"\nEpoch {epoch}/{EPOCHS} | Train={tr_acc:.2f}%", flush=True)
    acc, frec, nrec = validate("holdout")
    print("  Priority negatives:", flush=True)
    test_priority_negatives()

    if acc >= 90.0:
        torch.save(model.state_dict(), MODEL_PATH)
        marker = "↑ improved" if acc > best_acc else "maintained"
        print(f"  ✅ Saved (acc={acc:.2f}% {marker})", flush=True)
        best_acc = max(best_acc, acc)
    elif acc < 85.0:
        print(f"  🛑 Stopping — acc={acc:.2f}% < 85%, not saving", flush=True)
        # Restore backup to prevent bad save
        shutil.copy2(BACKUP_PATH, MODEL_PATH)
        print(f"  Restored backup.", flush=True)
        break
    else:
        print(f"  ⚠️  Acc={acc:.2f}% (90% threshold not met) — not saving", flush=True)

print(f"\n── Summary ─────────────────────────────────────────────────────────────", flush=True)
print(f"Baseline: acc={base_acc:.2f}%  flood-recall={base_frec:.2f}%  normal-recall={base_nrec:.2f}%", flush=True)
print(f"Best:     acc={best_acc:.2f}%", flush=True)
print(f"Model:    {MODEL_PATH}", flush=True)
