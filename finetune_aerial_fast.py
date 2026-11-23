"""
Phase 5 (Fast): Targeted fine-tuning for nadir aerial flood images.
Optimised for CPU: single forward pass, batch=64, 3 epochs only.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random
import shutil

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, "models", "flood_mobilenet_best.pt")
BACKUP_PATH = os.path.join(BASE, "models", "flood_mobilenet_backup.pt.bak")
FLOODNET    = os.path.join(BASE, "data", "FloodNet", "FloodNet-Supervised_v1.0")
ALLEY       = os.path.join(BASE, "data", "AlleyFloodNet")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}", flush=True)

# ── FloodNet label helper ─────────────────────────────────────────────────────
FLOOD_CLASSES = {1, 3, 5}

def get_floodnet_label(mask_path):
    try:
        mask = Image.open(mask_path)
        colors = mask.getcolors(maxcolors=50)
        if colors is None:
            return 0  # Water-heavy -> flood (Class 0)
        pixel_classes = {c[1] if isinstance(c[1], int) else c[1][0] for c in colors}
        return 0 if pixel_classes & FLOOD_CLASSES else 1  # 0: Flood, 1: Normal
    except Exception:
        return -1

# ── Collect FloodNet train ────────────────────────────────────────────────────
print("Collecting FloodNet labels…", flush=True)
img_dir  = os.path.join(FLOODNET, "train", "train-org-img")
mask_dir = os.path.join(FLOODNET, "train", "train-label-img")
fn_labeled = []
if os.path.exists(img_dir) and os.path.exists(mask_dir):
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_p = os.path.join(img_dir, fname)
        stem  = fname.rsplit(".", 1)[0]
        mask_p = os.path.join(mask_dir, stem + "_lab.png")
        if not os.path.exists(mask_p):
            mask_p = os.path.join(mask_dir, stem + ".png")
        if os.path.exists(mask_p):
            lbl = get_floodnet_label(mask_p)
            if lbl >= 0:
                fn_labeled.append((img_p, lbl))

flood_fn  = sum(1 for _, l in fn_labeled if l == 0)
normal_fn = sum(1 for _, l in fn_labeled if l == 1)
print(f"  FloodNet: {len(fn_labeled)} | Flood={flood_fn} Normal={normal_fn}", flush=True)

# ── Collect AlleyFloodNet ─────────────────────────────────────────────────────
def collect_alley(split_prefix):
    items = []
    for lbl, folder_name in [(0, "flooded"), (1, "non_flooded")]:
        for suffix in ["", "_"]:
            folder = os.path.join(ALLEY, split_prefix + suffix, folder_name)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        items.append((os.path.join(folder, f), lbl))
                break
    return items

alley_train = collect_alley("flood_train")
alley_val   = collect_alley("flood_val")
if not alley_val:
    alley_val = collect_alley("flood_test")
alley_val = alley_val[:200]

print(f"  AlleyFloodNet: train={len(alley_train)}, val={len(alley_val)}", flush=True)

all_train = fn_labeled + alley_train
val_items = alley_val
print(f"Total: train={len(all_train)}, val={len(val_items)}", flush=True)

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(45),
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
val_ds   = FloodDS(val_items, val_tf)

labels_list  = [l for _, l in all_train]
class_counts = [labels_list.count(0), labels_list.count(1)]
weights      = [1.0 / class_counts[l] for l in labels_list]
sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# batch=64 — fewer iterations per epoch
train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,    num_workers=0)

print(f"Batches per epoch: {len(train_loader)}", flush=True)

# ── Model ─────────────────────────────────────────────────────────────────────
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

# Freeze all, unfreeze features[14:] + classifier
for p in model.parameters():
    p.requires_grad = False
for layer in list(model.features[14:]) + list(model.classifier):
    for p in layer.parameters():
        p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}", flush=True)

shutil.copy2(MODEL_PATH, BACKUP_PATH)
print(f"Backup saved.", flush=True)

# ── Training (3 epochs, single forward pass per batch) ───────────────────────
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5, weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)

best_val_acc = 0.0
EPOCHS = 3

print(f"\nTraining — {EPOCHS} epochs", flush=True)
print("=" * 60, flush=True)

for epoch in range(1, EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    tr_correct = tr_total = 0
    for batch_idx, (imgs, lbls) in enumerate(train_loader):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        # reuse same `out` — NO second forward pass
        tr_correct += out.detach().argmax(1).eq(lbls).sum().item()
        tr_total   += lbls.size(0)
        if (batch_idx + 1) % 10 == 0:
            print(f"  E{epoch} batch {batch_idx+1}/{len(train_loader)}", flush=True)

    # ── Validate ───────────────────────────────────────────────────────────
    model.eval()
    val_correct = val_total = flood_correct = flood_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            preds = model(imgs).argmax(1)
            val_correct += preds.eq(lbls).sum().item()
            val_total   += lbls.size(0)
            fm = lbls == 1
            flood_correct += preds[fm].eq(1).sum().item()
            flood_total   += fm.sum().item()

    tr_acc  = 100 * tr_correct  / tr_total
    val_acc = 100 * val_correct / val_total
    recall  = 100 * flood_correct / flood_total if flood_total > 0 else 0.0
    scheduler.step()

    print(f"Epoch {epoch}/{EPOCHS} | Train={tr_acc:.2f}% | Val={val_acc:.2f}% | Flood-Recall={recall:.2f}%", flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✅ Saved best model (val={val_acc:.2f}%)", flush=True)

print(f"\nDone. Best val acc: {best_val_acc:.2f}%", flush=True)

