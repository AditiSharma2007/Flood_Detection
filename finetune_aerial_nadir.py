"""
Phase 5: Targeted Fine-tuning for Nadir (Top-Down) Aerial Flood Images
Focuses on FloodNet images where wide-area water coverage is dominant.
Addresses the failure mode: model classifying flooded aerial images as Normal.
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
import glob

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, "models", "flood_mobilenet_best.pt")
BACKUP_PATH = os.path.join(BASE, "models", "flood_mobilenet_backup.pt.bak")
FLOODNET    = os.path.join(BASE, "data", "FloodNet", "FloodNet-Supervised_v1.0")
ALLEY       = os.path.join(BASE, "data", "AlleyFloodNet")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ── Flood label helper (FloodNet) ─────────────────────────────────────────────
FLOOD_CLASSES = {1, 3, 5}  # building-flooded, road-flooded, water

def get_floodnet_label(mask_path):
    """Return 0 (flood) or 1 (normal) based on mask pixel classes."""
    try:
        mask = Image.open(mask_path)
        colors = mask.getcolors(maxcolors=50)
        if colors is None:
            return 0  # too many colors → water-heavy → flood (Class 0)
        pixel_classes = {c[1] if isinstance(c[1], int) else c[1][0] for c in colors}
        return 0 if pixel_classes & FLOOD_CLASSES else 1
    except Exception:
        return -1

# ── Build FloodNet image list ─────────────────────────────────────────────────
def collect_floodnet(split):
    img_dir  = os.path.join(FLOODNET, split, f"{split}-org-img")
    mask_dir = os.path.join(FLOODNET, split, f"{split}-label-img")
    if not os.path.exists(img_dir):
        return []
    pairs = []
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_p  = os.path.join(img_dir, fname)
        mask_p = os.path.join(mask_dir, fname.rsplit(".", 1)[0] + "_lab.png")
        if not os.path.exists(mask_p):
            # try alternate naming
            mask_p = os.path.join(mask_dir, fname.rsplit(".", 1)[0] + ".png")
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))
    return pairs

print("Collecting FloodNet labels (train + val)…")
fn_train = collect_floodnet("train")
fn_val   = collect_floodnet("val")
fn_all   = fn_train + fn_val
print(f"  FloodNet pairs found: {len(fn_all)}")

# Label each pair
fn_labeled = []
flood_count = normal_count = 0
for img_p, mask_p in fn_all:
    lbl = get_floodnet_label(mask_p)
    if lbl < 0:
        continue
    fn_labeled.append((img_p, lbl))
    if lbl == 0:
        flood_count += 1
    else:
        normal_count += 1
print(f"  Flood: {flood_count}, Normal: {normal_count}")

# ── Build AlleyFloodNet image list ────────────────────────────────────────────
def collect_alley(split_prefix):
    """Collect from AlleyFloodNet train/test splits."""
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
# Fall back: use test set for val if no val split
if len(alley_val) == 0:
    alley_val = collect_alley("flood_test")

print(f"  AlleyFloodNet train: {len(alley_train)}, val: {len(alley_val)}")

# ── Combined dataset ───────────────────────────────────────────────────────────
# Use ALL FloodNet (heavily upweighted) + AlleyFloodNet train
all_train_items = fn_labeled + alley_train
# Use AlleyFloodNet val for stability monitoring
val_items       = alley_val[:200] if len(alley_val) > 200 else alley_val

print(f"Total train samples: {len(all_train_items)}, val samples: {len(val_items)}")

# ── Transforms ────────────────────────────────────────────────────────────────
# Enhanced augmentation emphasising aerial perspective diversity
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Dataset class ─────────────────────────────────────────────────────────────
class FloodDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                return self.transform(img), label
        except Exception:
            # Return black image on error
            return torch.zeros(3, 224, 224), label

train_ds = FloodDataset(all_train_items, train_transform)
val_ds   = FloodDataset(val_items,       val_transform)

# Weighted sampler to balance classes
labels_arr = [lbl for _, lbl in all_train_items]
class_counts = [labels_arr.count(0), labels_arr.count(1)]
weights = [1.0 / class_counts[l] for l in labels_arr]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=24, sampler=sampler, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=24, shuffle=False,   num_workers=0, pin_memory=False)

# ── Load model ────────────────────────────────────────────────────────────────
def build_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(1280, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE)

model = build_model()

# ── Freeze strategy: unfreeze last 6 blocks + classifier ──────────────────────
# MobileNetV2 features: blocks 0-18 (inverted residuals)
# Unfreeze blocks 13-18 + classifier (more layers than Phase 4)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze features[13:] (blocks 13–18) and classifier
for layer in list(model.features[13:]) + list(model.classifier):
    for param in layer.parameters():
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5,          # lower LR to preserve existing knowledge
    weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)

# ── Backup current weights ────────────────────────────────────────────────────
import shutil
shutil.copy2(MODEL_PATH, BACKUP_PATH)
print(f"Backup saved to: {BACKUP_PATH}")

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0
EPOCHS = 5

print(f"\nStarting Phase 5 — {EPOCHS} epochs, {len(train_ds)} train / {len(val_ds)} val")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_correct = train_total = 0
    train_loss_sum = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        preds = out.argmax(1)
        train_correct += (preds == lbls).sum().item()
        train_total   += lbls.size(0)
        train_loss_sum += loss.item() * lbls.size(0)

    train_acc  = 100 * train_correct / train_total
    train_loss = train_loss_sum / train_total

    # Validate
    model.eval()
    val_correct = val_total = 0
    flood_correct = flood_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out   = model(imgs)
            preds = out.argmax(1)
            val_correct += (preds == lbls).sum().item()
            val_total   += lbls.size(0)
            flood_mask = (lbls == 1)
            flood_correct += (preds[flood_mask] == 1).sum().item()
            flood_total   += flood_mask.sum().item()

    val_acc      = 100 * val_correct / val_total
    flood_recall = 100 * flood_correct / flood_total if flood_total > 0 else 0.0

    scheduler.step()

    print(f"Epoch {epoch}/{EPOCHS} | Loss={train_loss:.4f} | Train={train_acc:.2f}% | Val={val_acc:.2f}% | Flood-Recall={flood_recall:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✅ Saved best model (val={val_acc:.2f}%)")

print(f"\nPhase 5 complete. Best val acc: {best_val_acc:.2f}%")
print(f"Model saved to: {MODEL_PATH}")

