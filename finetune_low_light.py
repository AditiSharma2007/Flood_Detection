"""
Fine-tune the production MobileNetV2 on low-brightness / low-contrast copies
of AlleyFloodNet so dusk, underexposed, and hazy photos classify more reliably.

Saves models/flood_mobilenet_best.pt only when:
  • original holdout accuracy stays >= 90%
  • synthetic dark/low-contrast holdout accuracy improves over baseline
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
from PIL import Image, ImageEnhance, ImageFilter

from lighting_utils import RandomLowLightContrast, enhance_lighting, IMAGENET_MEAN, IMAGENET_STD

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "flood_mobilenet_best.pt")
BACKUP_PATH = os.path.join(BASE, "models", "flood_mobilenet_backup.pt.bak")
ALLEY = os.path.join(BASE, "data", "AlleyFloodNet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOOD_LABEL = 0
NORMAL_LABEL = 1

print(f"Device: {DEVICE}", flush=True)


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
alley_val = collect_alley("flood_test_")
print(f"AlleyFloodNet: train={len(alley_train)}, holdout={len(alley_val)}", flush=True)

HARD_NEGATIVES = [
    os.path.join(BASE, "test_images", name)
    for name in (
        "sunset_lake.jpg",
        "city_river.jpg",
        "waterfall_lagoon.jpg",
        "eiffel_tower.jpg",
        "user_normal_river.jpg",
        "user_code_screenshot.png",
    )
]
valid_hn = [(p, NORMAL_LABEL) for p in HARD_NEGATIVES if os.path.exists(p)]
for p, _ in valid_hn:
    print(f"  Keep hard-neg: {os.path.basename(p)}", flush=True)

all_train = alley_train + valid_hn * 8
n_f = sum(1 for _, l in all_train if l == FLOOD_LABEL)
n_n = sum(1 for _, l in all_train if l == NORMAL_LABEL)
print(f"Train: {len(all_train)} (Flood={n_f}, Normal={n_n})", flush=True)

train_tf = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        RandomLowLightContrast(p=0.65),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.12),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
val_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def degrade_for_eval(img: Image.Image) -> Image.Image:
    """Fixed (non-random) dusk + low-contrast transform for a stable dark val set."""
    out = ImageEnhance.Brightness(img.convert("RGB")).enhance(0.42)
    lut = [min(255, int(round(255.0 * ((i / 255.0) ** 1.55)))) for i in range(256)]
    out = out.point(lut * 3)
    out = ImageEnhance.Contrast(out).enhance(0.55)
    veil = Image.new("RGB", out.size, (165, 165, 170))
    out = Image.blend(out, veil, 0.18)
    return out.filter(ImageFilter.GaussianBlur(radius=0.35))


class FloodDS(Dataset):
    def __init__(self, items, tf, degrade=False, restore=False):
        self.items = items
        self.tf = tf
        self.degrade = degrade
        self.restore = restore

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            if self.degrade:
                rgb = degrade_for_eval(rgb)
            if self.restore:
                rgb = enhance_lighting(rgb)
            return self.tf(rgb), lbl
        except Exception:
            return torch.zeros(3, 224, 224), lbl


train_ds = FloodDS(all_train, train_tf)
val_ds = FloodDS(alley_val, val_tf)
dark_val_ds = FloodDS(alley_val, val_tf, degrade=True)
dark_restored_ds = FloodDS(alley_val, val_tf, degrade=True, restore=True)

labels_list = [l for _, l in all_train]
cc = [labels_list.count(0), labels_list.count(1)]
weights = [1.0 / cc[l] for l in labels_list]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
dark_loader = DataLoader(dark_val_ds, batch_size=32, shuffle=False, num_workers=0)
restored_loader = DataLoader(dark_restored_ds, batch_size=32, shuffle=False, num_workers=0)


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
for p in model.parameters():
    p.requires_grad = False
for layer in list(model.features[17:]) + list(model.classifier):
    for p in layer.parameters():
        p.requires_grad = True

print(
    f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    flush=True,
)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
if os.path.exists(MODEL_PATH):
    shutil.copy2(MODEL_PATH, BACKUP_PATH)
    print("Backup saved.", flush=True)


@torch.no_grad()
def validate(loader, tag="val"):
    model.eval()
    c = t = fc = ft = nc = nt = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        preds = model(imgs).argmax(1)
        c += preds.eq(lbls).sum().item()
        t += lbls.size(0)
        fm = lbls == 0
        fc += preds[fm].eq(0).sum().item()
        ft += fm.sum().item()
        nm = lbls == 1
        nc += preds[nm].eq(1).sum().item()
        nt += nm.sum().item()
    acc = 100 * c / t if t else 0
    fr = 100 * fc / ft if ft else 0
    nr = 100 * nc / nt if nt else 0
    print(
        f"  [{tag}] Acc={acc:.2f}%  Flood-Recall={fr:.2f}%  Normal-Recall={nr:.2f}%",
        flush=True,
    )
    return acc


def test_hard_negs():
    if not valid_hn:
        return True
    model.eval()
    n_correct = 0
    for path, exp in valid_hn:
        with Image.open(path) as img:
            t = val_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1)[0]
        pred = 0 if probs[0].item() > 0.5 else 1
        ok = pred == exp
        n_correct += int(ok)
        icon = "✅" if ok else "🚨"
        print(
            f"    {icon} Flood={probs[0].item()*100:.1f}% | {os.path.basename(path)}",
            flush=True,
        )
    print(f"    → {n_correct}/{len(valid_hn)} hard-neg correct", flush=True)
    return n_correct == len(valid_hn)


print("\n── Baseline ────────────────────────────────────────────────", flush=True)
base_orig = validate(val_loader, "holdout")
base_dark = validate(dark_loader, "dark-holdout")
base_rest = validate(restored_loader, "dark+CLAHE")
test_hard_negs()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    weight_decay=1e-3,
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
EPOCHS = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

best_dark = base_dark
best_orig = base_orig

print(f"\n── Training ({EPOCHS} epochs) ──────────────────────────────", flush=True)
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

    print(f"\nEpoch {epoch}/{EPOCHS} | Train={100 * tc / tt:.2f}%", flush=True)
    orig_acc = validate(val_loader, "holdout")
    dark_acc = validate(dark_loader, "dark-holdout")
    validate(restored_loader, "dark+CLAHE")
    test_hard_negs()

    improved_dark = dark_acc > best_dark + 0.05
    orig_ok = orig_acc >= 90.0 and orig_acc >= (base_orig - 1.5)
    if orig_ok and (improved_dark or dark_acc >= best_dark) and orig_acc >= best_orig - 0.5:
        torch.save(model.state_dict(), MODEL_PATH)
        best_dark = max(best_dark, dark_acc)
        best_orig = max(best_orig, orig_acc)
        print(
            f"  ✅ Saved (orig={orig_acc:.2f}%, dark={dark_acc:.2f}%)",
            flush=True,
        )
    elif orig_acc < 85.0:
        print(f"  🛑 Restoring backup — orig acc too low ({orig_acc:.2f}%)", flush=True)
        shutil.copy2(BACKUP_PATH, MODEL_PATH)
        model = build_model()
        break
    else:
        print(
            f"  ⚠️ Not saved (orig={orig_acc:.2f}%, dark={dark_acc:.2f}%)",
            flush=True,
        )

print("\n── Summary ─────────────────────────────────────────────────", flush=True)
print(
    f"Original holdout: {base_orig:.2f}% → best {best_orig:.2f}%",
    flush=True,
)
print(
    f"Dark holdout:     {base_dark:.2f}% → best {best_dark:.2f}%",
    flush=True,
)
print(f"Checkpoint: {MODEL_PATH}", flush=True)
