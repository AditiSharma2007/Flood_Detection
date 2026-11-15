"""
Phase 5 (Conservative): Classifier-only fine-tuning for aerial flood images.
- ONLY the final classifier (Linear layers) are updated — features are FROZEN
- Very low LR (1e-5) to avoid destabilising existing decision boundary
- Validates on BOTH AlleyFloodNet ground + a sample of FloodNet aerial
- Saves only if BOTH val sets improve
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
            return 1
        pixel_classes = {c[1] if isinstance(c[1], int) else c[1][0] for c in colors}
        return 1 if pixel_classes & FLOOD_CLASSES else 0
    except Exception:
        return -1

# ── Collect FloodNet train ────────────────────────────────────────────────────
print("Collecting FloodNet labels…", flush=True)
img_dir  = os.path.join(FLOODNET, "train", "train-org-img")
mask_dir = os.path.join(FLOODNET, "train", "train-label-img")
fn_labeled = []
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

flood_fn  = sum(1 for _, l in fn_labeled if l == 1)
normal_fn = sum(1 for _, l in fn_labeled if l == 0)
print(f"  FloodNet train: {len(fn_labeled)} | Flood={flood_fn} Normal={normal_fn}", flush=True)

# Separate FloodNet into train (80%) and aerial-val (20%)
random.shuffle(fn_labeled)
split = int(0.8 * len(fn_labeled))
fn_train = fn_labeled[:split]
fn_val   = fn_labeled[split:]   # aerial validation set
print(f"  FloodNet split: train={len(fn_train)}, aerial-val={len(fn_val)}", flush=True)

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

print(f"  AlleyFloodNet: train={len(alley_train)}, ground-val={len(alley_val)}", flush=True)

# ── Datasets ──────────────────────────────────────────────────────────────────
# Train: FloodNet aerial (80%) + AlleyFloodNet ground
all_train  = fn_train + alley_train
# Val: FloodNet aerial (20%) — aerial performance monitoring
# Val: AlleyFloodNet ground — ground accuracy preservation
ground_val = alley_val[:200]
aerial_val = fn_val[:100]  # 100 aerial val images

print(f"Total: train={len(all_train)}, ground-val={len(ground_val)}, aerial-val={len(aerial_val)}", flush=True)

# ── Transforms — MILD augmentation to avoid distribution shift ────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),   # mild crop only
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),        # subtle jitter
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

train_ds     = FloodDS(all_train,  train_tf)
ground_val_ds = FloodDS(ground_val, val_tf)
aerial_val_ds = FloodDS(aerial_val, val_tf)

labels_list  = [l for _, l in all_train]
class_counts = [labels_list.count(0), labels_list.count(1)]
weights      = [1.0 / class_counts[l] for l in labels_list]
sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader       = DataLoader(train_ds,      batch_size=64, sampler=sampler, num_workers=0)
ground_val_loader  = DataLoader(ground_val_ds, batch_size=64, shuffle=False,   num_workers=0)
aerial_val_loader  = DataLoader(aerial_val_ds, batch_size=64, shuffle=False,   num_workers=0)

print(f"Batches per epoch: {len(train_loader)}", flush=True)

# ── Model — load fresh, freeze ALL features, only train classifier ────────────
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

# ── FREEZE ALL FEATURES — classifier only ────────────────────────────────────
for p in model.features.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} (classifier only)", flush=True)

shutil.copy2(MODEL_PATH, BACKUP_PATH)
print(f"Backup saved.", flush=True)

# ── Validate helper ───────────────────────────────────────────────────────────
def validate(loader, label="val"):
    model.eval()
    correct = total = flood_correct = flood_total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += preds.eq(lbls).sum().item()
            total   += lbls.size(0)
            fm = lbls == 1
            flood_correct += preds[fm].eq(1).sum().item()
            flood_total   += fm.sum().item()
    acc    = 100 * correct / total if total > 0 else 0
    recall = 100 * flood_correct / flood_total if flood_total > 0 else 0
    print(f"  [{label}] Acc={acc:.2f}%  Flood-Recall={recall:.2f}%  ({total} samples)", flush=True)
    return acc, recall

# Baseline before any training
print("\n── Baseline (pre-training) ──", flush=True)
base_ground_acc, base_ground_recall = validate(ground_val_loader, "ground")
base_aerial_acc, base_aerial_recall = validate(aerial_val_loader, "aerial")

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,              # very conservative LR
    weight_decay=1e-3
)
criterion  = nn.CrossEntropyLoss(label_smoothing=0.05)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

best_ground_acc = base_ground_acc
best_aerial_acc = base_aerial_acc
EPOCHS = 5

print(f"\n── Training ({EPOCHS} epochs, classifier only) ──", flush=True)
print("=" * 60, flush=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_correct = tr_total = 0
    for batch_idx, (imgs, lbls) in enumerate(train_loader):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        tr_correct += out.detach().argmax(1).eq(lbls).sum().item()
        tr_total   += lbls.size(0)
        if (batch_idx + 1) % 5 == 0:
            print(f"  E{epoch} batch {batch_idx+1}/{len(train_loader)}", flush=True)

    tr_acc = 100 * tr_correct / tr_total
    print(f"\nEpoch {epoch}/{EPOCHS} | Train={tr_acc:.2f}%", flush=True)
    ground_acc, ground_recall = validate(ground_val_loader, "ground")
    aerial_acc, aerial_recall = validate(aerial_val_loader, "aerial")
    scheduler.step()

    # Save only if ground accuracy is preserved (≥90%) AND aerial improves
    if ground_acc >= 90.0 and aerial_acc > best_aerial_acc:
        best_ground_acc = ground_acc
        best_aerial_acc = aerial_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✅ Saved (ground={ground_acc:.2f}% ≥90%, aerial={aerial_acc:.2f}%↑)", flush=True)
    elif ground_acc < 90.0:
        print(f"  ⚠️  Ground acc dropped to {ground_acc:.2f}% — NOT saving", flush=True)
    else:
        print(f"  — No improvement in aerial acc ({aerial_acc:.2f}% vs best {best_aerial_acc:.2f}%)", flush=True)

print(f"\n── Summary ──────────────────────────────────────────────", flush=True)
print(f"Baseline:  ground={base_ground_acc:.2f}%  aerial={base_aerial_acc:.2f}%", flush=True)
print(f"Best:      ground={best_ground_acc:.2f}%  aerial={best_aerial_acc:.2f}%", flush=True)
print(f"Model: {MODEL_PATH}", flush=True)


