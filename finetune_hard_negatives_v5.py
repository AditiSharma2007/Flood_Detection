"""
Hard-Negative Calibration v5 — Fix false positives on normal water images
Uses 6 user-provided Normal images (lakes, rivers, waterfalls, cityscapes, screenshots)
as hard negatives with heavy oversampling + backbone fine-tuning.

Label convention: Class 0 = Flooding, Class 1 = Normal
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
random.seed(42); np.random.seed(42); torch.manual_seed(42)

BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, "models", "flood_mobilenet_best.pt")
BACKUP_PATH = os.path.join(BASE, "models", "flood_mobilenet_backup.pt.bak")
ALLEY       = os.path.join(BASE, "data", "AlleyFloodNet")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOOD_LABEL = 0; NORMAL_LABEL = 1

print(f"Device: {DEVICE}", flush=True)

# ── All 6 hard negatives (Normal images the model gets wrong) ─────────────────
HARD_NEGATIVES = [
    "test_images/sunset_lake.jpg",          # scenic sunset over lake
    "test_images/city_river.jpg",           # aerial city with river
    "test_images/waterfall_lagoon.jpg",     # waterfall + tropical lagoon
    "test_images/eiffel_tower.jpg",         # Eiffel Tower + Seine River
    "test_images/user_normal_river.jpg",    # rural river with green grass
    "test_images/user_code_screenshot.png", # code screenshot (non-photo)
]

valid_hn = []
for rel in HARD_NEGATIVES:
    p = os.path.join(BASE, rel)
    if os.path.exists(p):
        valid_hn.append((p, NORMAL_LABEL))
        print(f"  Hard-neg: {os.path.basename(p)}", flush=True)

# ── Collect AlleyFloodNet ─────────────────────────────────────────────────────
def collect_alley(split_dir):
    items = []
    for lbl, folder in [(FLOOD_LABEL, "flooded"), (NORMAL_LABEL, "non_flooded")]:
        d = os.path.join(ALLEY, split_dir, folder)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    items.append((os.path.join(d, f), lbl))
    return items

alley_train = collect_alley("flood_train_")
alley_val   = collect_alley("flood_test_")
print(f"AlleyFloodNet: train={len(alley_train)}, holdout={len(alley_val)}", flush=True)

# Oversample: 6 images × 20 copies = 120 hard neg samples in the train set
OVERSAMPLE = 20
all_train = alley_train + valid_hn * OVERSAMPLE
n_f = sum(1 for _, l in all_train if l == FLOOD_LABEL)
n_n = sum(1 for _, l in all_train if l == NORMAL_LABEL)
print(f"Train: {len(all_train)} (Flood={n_f}, Normal={n_n})", flush=True)

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
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
        self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        try:
            with Image.open(path) as img:
                return self.tf(img.convert("RGB")), lbl
        except Exception:
            return torch.zeros(3, 224, 224), lbl

train_ds = FloodDS(all_train, train_tf)
val_ds   = FloodDS(alley_val, val_tf)

labels_list = [l for _, l in all_train]
cc = [labels_list.count(0), labels_list.count(1)]
weights = [1.0/cc[l] for l in labels_list]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,   num_workers=0)
print(f"Batches/epoch: {len(train_loader)}", flush=True)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(1280, 64),
        nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(64, 2),
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE)

model = build_model()

# Unfreeze features[17:] + classifier (same strategy that worked in v4)
for p in model.parameters():
    p.requires_grad = False
for layer in list(model.features[17:]) + list(model.classifier):
    for p in layer.parameters():
        p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,} (features[17:] + classifier)", flush=True)

shutil.copy2(MODEL_PATH, BACKUP_PATH)
print("Backup saved.", flush=True)

# ── Validation ────────────────────────────────────────────────────────────────
def validate(tag="val"):
    model.eval()
    c = t = fc = ft = nc = nt = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            preds = model(imgs).argmax(1)
            c += preds.eq(lbls).sum().item(); t += lbls.size(0)
            fm = lbls == 0; fc += preds[fm].eq(0).sum().item(); ft += fm.sum().item()
            nm = lbls == 1; nc += preds[nm].eq(1).sum().item(); nt += nm.sum().item()
    acc = 100*c/t; fr = 100*fc/ft if ft else 0; nr = 100*nc/nt if nt else 0
    print(f"  [{tag}] Acc={acc:.2f}%  Flood-Recall={fr:.2f}%  Normal-Recall={nr:.2f}%", flush=True)
    return acc

def test_hard_negs():
    model.eval()
    n_correct = 0
    for path, exp in valid_hn:
        with Image.open(path) as img:
            t = val_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1)[0]
        fp, np_ = probs[0].item()*100, probs[1].item()*100
        pred = 0 if fp > 50 else 1
        ok = pred == exp
        if ok: n_correct += 1
        icon = "✅" if ok else "🚨"
        print(f"    {icon} Flood={fp:.1f}% Normal={np_:.1f}% | {os.path.basename(path)}", flush=True)
    print(f"    → {n_correct}/{len(valid_hn)} correct", flush=True)
    return n_correct == len(valid_hn)

# ── Baseline ──────────────────────────────────────────────────────────────────
print("\n── Baseline ────────────────────────────────────────────────", flush=True)
base_acc = validate("holdout")
test_hard_negs()

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=8e-6, weight_decay=1e-3
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-7)

best_acc = base_acc
EPOCHS = 12

print(f"\n── Training ({EPOCHS} epochs) ──────────────────────────────────", flush=True)
print("=" * 65, flush=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    tc = tt = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        tc += out.detach().argmax(1).eq(lbls).sum().item()
        tt += lbls.size(0)
    scheduler.step()

    tr_acc = 100*tc/tt
    print(f"\nEpoch {epoch}/{EPOCHS} | Train={tr_acc:.2f}%", flush=True)
    acc = validate("holdout")
    all_ok = test_hard_negs()

    if acc >= 90.0:
        torch.save(model.state_dict(), MODEL_PATH)
        tag = "↑" if acc > best_acc else "="
        best_acc = max(best_acc, acc)
        print(f"  ✅ Saved (acc={acc:.2f}%{tag})", flush=True)
        if all_ok:
            print(f"  🎯 ALL {len(valid_hn)} hard negatives fixed! Stopping.", flush=True)
            break
    elif acc < 85.0:
        print(f"  🛑 Restoring backup — acc too low ({acc:.2f}%)", flush=True)
        shutil.copy2(BACKUP_PATH, MODEL_PATH)
        break
    else:
        print(f"  ⚠️ Acc={acc:.2f}% — not saving", flush=True)

print(f"\n── Summary ─────────────────────────────────────────────────", flush=True)
print(f"Baseline: {base_acc:.2f}%  →  Best: {best_acc:.2f}%", flush=True)
