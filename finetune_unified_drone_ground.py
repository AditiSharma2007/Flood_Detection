import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import random
import shutil
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLOODNET_DIR = os.path.join(BASE_DIR, "data", "FloodNet", "FloodNet-Supervised_v1.0")
ALLEY_DIR = os.path.join(BASE_DIR, "data", "AlleyFloodNet")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(BASE_DIR, "data", "train_cache")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "flood_mobilenet_best.pt")
OUTPUT_METRICS_PATH = os.path.join(MODELS_DIR, "improved_benchmark_comparison.json")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"UNIFIED GROUND & DRONE ACCURACY ENHANCEMENT PIPELINE")
print(f"Device:           {device}")
print(f"Base Model:       {BEST_MODEL_PATH}")
print(f"Cache Directory:  {CACHE_DIR}")
print("=" * 70)

# ==============================================================================
# 1. Fast Dataset Pre-Caching
# ==============================================================================
def prepare_cache():
    cache_train_flood = os.path.join(CACHE_DIR, "train", "flooded")
    cache_train_normal = os.path.join(CACHE_DIR, "train", "normal")
    cache_val_flood = os.path.join(CACHE_DIR, "val", "flooded")
    cache_val_normal = os.path.join(CACHE_DIR, "val", "normal")

    for d in [cache_train_flood, cache_train_normal, cache_val_flood, cache_val_normal]:
        os.makedirs(d, exist_ok=True)

    # Check if already cached
    n_existing_tr = len(os.listdir(cache_train_flood)) + len(os.listdir(cache_train_normal))
    if n_existing_tr > 2000:
        print(f"[OK] Cache already populated ({n_existing_tr} training images). Skipping generation.")
        return cache_train_flood, cache_train_normal, cache_val_flood, cache_val_normal

    print("[*] Generating fast 256x256 training cache for FloodNet & AlleyFloodNet...")
    t0 = time.time()

    # 1.1 Cache FloodNet Train
    fn_tr_org = os.path.join(FLOODNET_DIR, "train", "train-org-img")
    fn_tr_lbl = os.path.join(FLOODNET_DIR, "train", "train-label-img")
    if os.path.exists(fn_tr_org):
        files = sorted(os.listdir(fn_tr_org))
        print(f"  Processing {len(files)} FloodNet Train images...")
        for fname in files:
            base_id = os.path.splitext(fname)[0]
            mask_name = f"{base_id}_lab.png"
            mask_path = os.path.join(fn_tr_lbl, mask_name)
            if not os.path.exists(mask_path):
                continue

            with Image.open(mask_path) as m_img:
                colors = m_img.getcolors(maxcolors=50)

            has_flood = False
            if colors:
                for cnt, val in colors:
                    if val in (1, 3, 5):
                        has_flood = True
                        break

            dest_dir = cache_train_flood if has_flood else cache_train_normal
            dest_file = os.path.join(dest_dir, f"fn_tr_{base_id}.jpg")

            if not os.path.exists(dest_file):
                try:
                    with Image.open(os.path.join(fn_tr_org, fname)) as img:
                        img.draft("RGB", (256, 256))
                        img_rgb = img.convert("RGB")
                        img_res = img_rgb.resize((256, 256), Image.Resampling.BILINEAR)
                        img_res.save(dest_file, quality=90)
                except Exception:
                    pass

    # 1.2 Cache FloodNet Val
    fn_val_org = os.path.join(FLOODNET_DIR, "val", "val-org-img")
    fn_val_lbl = os.path.join(FLOODNET_DIR, "val", "val-label-img")
    if os.path.exists(fn_val_org):
        files = sorted(os.listdir(fn_val_org))
        print(f"  Processing {len(files)} FloodNet Val images...")
        for fname in files:
            base_id = os.path.splitext(fname)[0]
            mask_name = f"{base_id}_lab.png"
            mask_path = os.path.join(fn_val_lbl, mask_name)
            if not os.path.exists(mask_path):
                continue

            with Image.open(mask_path) as m_img:
                colors = m_img.getcolors(maxcolors=50)

            has_flood = False
            if colors:
                for cnt, val in colors:
                    if val in (1, 3, 5):
                        has_flood = True
                        break

            dest_dir = cache_val_flood if has_flood else cache_val_normal
            dest_file = os.path.join(dest_dir, f"fn_val_{base_id}.jpg")

            if not os.path.exists(dest_file):
                try:
                    with Image.open(os.path.join(fn_val_org, fname)) as img:
                        img.draft("RGB", (256, 256))
                        img_rgb = img.convert("RGB")
                        img_res = img_rgb.resize((256, 256), Image.Resampling.BILINEAR)
                        img_res.save(dest_file, quality=90)
                except Exception:
                    pass

    # 1.3 Cache AlleyFloodNet Train (Urban Ground domain)
    alley_fld = os.path.join(ALLEY_DIR, "flood_train_", "flooded")
    alley_norm = os.path.join(ALLEY_DIR, "flood_train_", "non_flooded")
    if os.path.exists(alley_fld):
        for f in os.listdir(alley_fld):
            src = os.path.join(alley_fld, f)
            dest = os.path.join(cache_train_flood, f"alley_{f}")
            if not os.path.exists(dest):
                try:
                    with Image.open(src) as img:
                        img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR).save(dest, quality=90)
                except Exception:
                    pass

    if os.path.exists(alley_norm):
        for f in os.listdir(alley_norm):
            src = os.path.join(alley_norm, f)
            dest = os.path.join(cache_train_normal, f"alley_{f}")
            if not os.path.exists(dest):
                try:
                    with Image.open(src) as img:
                        img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR).save(dest, quality=90)
                except Exception:
                    pass

    print(f"[OK] Cache generated in {time.time()-t0:.1f}s.")
    return cache_train_flood, cache_train_normal, cache_val_flood, cache_val_normal

# ==============================================================================
# 2. Dataset & Transforms
# ==============================================================================
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FastFolderDataset(Dataset):
    def __init__(self, flood_dir, normal_dir, transform=None):
        self.samples = []
        self.transform = transform

        for f in os.listdir(flood_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(flood_dir, f), 0))

        for f in os.listdir(normal_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(normal_dir, f), 1))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, label

# ==============================================================================
# 3. Model Architecture & Fine-Tuning Setup
# ==============================================================================
def create_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2)
    )
    state = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    # Freeze base feature extractor, unfreeze top blocks (14-18) + classifier
    for param in model.features[:14].parameters():
        param.requires_grad = False
    for param in model.features[14:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

# ==============================================================================
# 4. Training Loop
# ==============================================================================
def train_model():
    cache_tr_f, cache_tr_n, cache_v_f, cache_v_n = prepare_cache()

    train_ds = FastFolderDataset(cache_tr_f, cache_tr_n, transform=train_transforms)
    val_ds = FastFolderDataset(cache_v_f, cache_v_n, transform=val_transforms)

    n_fld = sum(1 for _, l in train_ds.samples if l == 0)
    n_nrm = sum(1 for _, l in train_ds.samples if l == 1)
    w_fld = len(train_ds) / (2.0 * n_fld)
    w_nrm = len(train_ds) / (2.0 * n_nrm)
    class_weights = torch.tensor([w_fld, w_nrm], dtype=torch.float).to(device)

    print(f"\nTraining Samples: {len(train_ds)} ({n_fld} Flooded, {n_nrm} Normal)")
    print(f"Validation Samples: {len(val_ds)}")
    print(f"Class Weights: Flood={w_fld:.3f}, Normal={w_nrm:.3f}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = create_model()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4
    )
    epochs = 4
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    print("\n" + "=" * 65)
    print("STARTING FINE-TUNING (4 Epochs)...")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        correct_tr = 0
        total_tr = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)
            correct_tr += (preds == labels).sum().item()
            total_tr += len(labels)

        scheduler.step()
        train_loss /= total_tr
        train_acc = correct_tr / total_tr

        # Validation
        model.eval()
        val_loss = 0.0
        correct_v = 0
        total_v = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * len(labels)
                preds = torch.argmax(outputs, dim=1)
                correct_v += (preds == labels).sum().item()
                total_v += len(labels)

        val_loss /= total_v
        val_acc = correct_v / total_v
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss or val_acc > best_val_acc:
            best_val_loss = min(val_loss, best_val_loss)
            best_val_acc = max(val_acc, best_val_acc)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  [SAVED] Checkpoint improved! Saved to {BEST_MODEL_PATH}")

    print("\n[OK] Training completed. Best checkpoint saved.")

# ==============================================================================
# 5. Independent Benchmark Evaluation (Holdout Test Sets)
# ==============================================================================
def evaluate_benchmarks():
    print("\n" + "=" * 65)
    print("RUNNING POST-TRAINING BENCHMARK EVALUATIONS...")
    print("=" * 65)

    # 5.1 Run test_model_on_floodnet.py
    import subprocess
    cmd = [sys.executable, os.path.join(BASE_DIR, "test_model_on_floodnet.py")]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    print("=== FLOODNET POST-TRAINING EVALUATION ===")
    print(p.stdout)
    if p.stderr:
        print("STDERR:", p.stderr)

    # 5.2 AlleyFloodNet Ground Benchmark
    alley_te_flood = os.path.join(ALLEY_DIR, "flood_test_", "flooded")
    alley_te_norm = os.path.join(ALLEY_DIR, "flood_test_", "non_flooded")

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2)
    )
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_imgs = []
    if os.path.exists(alley_te_flood):
        for f in os.listdir(alley_te_flood):
            test_imgs.append((os.path.join(alley_te_flood, f), 0))
    if os.path.exists(alley_te_norm):
        for f in os.listdir(alley_te_norm):
            test_imgs.append((os.path.join(alley_te_norm, f), 1))

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for p_img, target in test_imgs:
            try:
                with Image.open(p_img) as img:
                    t = val_transforms(img.convert("RGB")).unsqueeze(0).to(device)
                    logits = model(t)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred = torch.argmax(probs).item()
                    y_true.append(target)
                    y_pred.append(pred)
                    y_probs.append(probs[0].item())
            except Exception:
                pass

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    alley_acc = accuracy_score(y_true, y_pred)
    alley_rec = recall_score(y_true, y_pred, pos_label=0)
    alley_prec = precision_score(y_true, y_pred, pos_label=0)
    alley_f1 = f1_score(y_true, y_pred, pos_label=0)
    alley_auc = roc_auc_score(1 - y_true, y_probs)

    print("\n=== ALLEYFLOODNET (GROUND DOMAIN) BENCHMARK ===")
    print(f"  Holdout Test Images: {len(test_imgs)}")
    print(f"  Accuracy:            {alley_acc * 100:.2f}%")
    print(f"  Flooding Recall:     {alley_rec * 100:.2f}%")
    print(f"  Flooding Precision:  {alley_prec * 100:.2f}%")
    print(f"  Flooding F1-Score:   {alley_f1 * 100:.2f}%")
    print(f"  ROC-AUC:             {alley_auc:.4f}")

    # Load FloodNet results JSON
    fn_json = os.path.join(BASE_DIR, "models", "floodnet_eval_results", "floodnet_evaluation_metrics.json")
    with open(fn_json, "r") as f:
        fn_metrics = json.load(f)

    comparison = {
        "floodnet_test": fn_metrics["test"],
        "floodnet_val": fn_metrics["val"],
        "floodnet_overall": fn_metrics["overall"],
        "alleyfloodnet_test": {
            "total_images": len(test_imgs),
            "accuracy": float(alley_acc),
            "flooding_recall": float(alley_rec),
            "flooding_precision": float(alley_prec),
            "flooding_f1": float(alley_f1),
            "roc_auc": float(alley_auc)
        }
    }

    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)
    print(f"\n[OK] Comparison report saved to: {OUTPUT_METRICS_PATH}")

if __name__ == "__main__":
    train_model()
    evaluate_benchmarks()

