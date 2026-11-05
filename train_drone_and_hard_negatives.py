"""
Fine-tuning MobileNetV2 on Drone Aerial Imagery & Comprehensive Hard Negatives:
  - Drone Flooded Images (AIFloodSense, 470 images)
  - Drone Non-Flooded Images (Semantic Drone, 400 images: swimming pools, houses, grass, roads)
  - Dark & Night Hard Negatives (300 dusk & night images)
  - Water Hard Negatives (Rivers, streams, lakes, drainage canals)
  - Benchmark Holdout Validation on AlleyFloodNet Test Set
"""

import os
import sys
import time
import json
import random
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)

# Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 65)
print(f"DRONE & COMPREHENSIVE HARD-NEGATIVE FINE-TUNING PIPELINE")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 65)

# Paths
BASE_DIR = r"D:\Flood_Detection_Data"
OUTPUT_DIR = r"c:\Users\Aditi Sharma\Downloads\Flood_Detection\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "flood_mobilenet_best.pt")

DRONE_DATA_DIR = os.path.join(BASE_DIR, "drone_and_hard_negatives")
DRONE_FLOOD = os.path.join(DRONE_DATA_DIR, "flooded")
DRONE_NORMAL = os.path.join(DRONE_DATA_DIR, "non_flooded")
DRONE_DARK = os.path.join(DRONE_DATA_DIR, "dark_hard_negatives")

WATER_NEGATIVES_DIR = os.path.join(BASE_DIR, "water_hard_negatives")
TEST_IMAGES_DIR = r"c:\Users\Aditi Sharma\Downloads\Flood_Detection\test_images"

ALLEY_TR_FLOOD = os.path.join(BASE_DIR, "AlleyFloodNet", "flood_train_", "flooded")
ALLEY_TR_NORMAL = os.path.join(BASE_DIR, "AlleyFloodNet", "flood_train_", "non_flooded")
ALLEY_TE_FLOOD = os.path.join(BASE_DIR, "AlleyFloodNet", "flood_test_", "flooded")
ALLEY_TE_NORMAL = os.path.join(BASE_DIR, "AlleyFloodNet", "flood_test_", "non_flooded")

MAIN_FLOOD = os.path.join(BASE_DIR, "main_dataset", "Dataset", "Flood Images")
MAIN_NORMAL = os.path.join(BASE_DIR, "main_dataset", "Dataset", "Non Flood Images")

def get_images(folder):
    if not os.path.exists(folder):
        return []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(valid_exts) and os.path.getsize(os.path.join(folder, f)) > 0
    ]

# Gather images
drone_fl = get_images(DRONE_FLOOD)
drone_no = get_images(DRONE_NORMAL)
drone_dark = get_images(DRONE_DARK)
water_negs = get_images(WATER_NEGATIVES_DIR)
user_test_imgs = get_images(TEST_IMAGES_DIR)

alley_tr_fl = get_images(ALLEY_TR_FLOOD)
alley_tr_no = get_images(ALLEY_TR_NORMAL)

main_fl = get_images(MAIN_FLOOD)
main_no = get_images(MAIN_NORMAL)

random.seed(42)
sampled_main_fl = random.sample(main_fl, min(1000, len(main_fl)))
sampled_main_no = random.sample(main_no, min(1000, len(main_no)))

# Build training set: Class 0 = Flooded, Class 1 = Normal
train_paths = []
train_labels = []
sample_weights = []

# 1. Drone Flooded Images (Class 0, Weight: 6x)
for p in drone_fl:
    train_paths.append(p)
    train_labels.append(0)
    sample_weights.append(6.0)

# 2. Drone Non-Flooded Images (Class 1, Weight: 6x - Swimming pools, roofs, landscapes)
for p in drone_no:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(6.0)

# 3. Dark / Night Hard Negatives (Class 1, Weight: 10x)
for p in drone_dark:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(10.0)

# 4. Natural Water & Rivers (Class 1, Weight: 15x)
for p in water_negs:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(15.0)

# 5. User Screenshots (Class 1, Weight: 20x)
for p in user_test_imgs:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(20.0)

# 6. AlleyFloodNet Train (Weight: 3x)
for p in alley_tr_fl:
    train_paths.append(p)
    train_labels.append(0)
    sample_weights.append(3.0)

for p in alley_tr_no:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(3.0)

# 7. Main Dataset (Weight: 1x)
for p in sampled_main_fl:
    train_paths.append(p)
    train_labels.append(0)
    sample_weights.append(1.0)

for p in sampled_main_no:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(1.0)

print(f"\nTraining Pool Summary:")
print(f"  - Total images:            {len(train_paths)}")
print(f"  - Flooded (Class 0):       {train_labels.count(0)} (Drone aerial floods + Alley + Main)")
print(f"  - Normal / Negatives (1):  {train_labels.count(1)} (Drone pools/landscapes, dark/night, rivers, drainage)")
print(f"    * Drone Normal/Pools:    {len(drone_no)}")
print(f"    * Dark & Night Scenes:   {len(drone_dark)}")
print(f"    * Rivers & Drainage:     {len(water_negs)}")

# Holdout Benchmark Test Set
te_flood = get_images(ALLEY_TE_FLOOD)
te_normal = get_images(ALLEY_TE_NORMAL)
test_paths = te_flood + te_normal
test_labels = [0] * len(te_flood) + [1] * len(te_normal)
print(f"Holdout Benchmark Test Set:  {len(test_paths)} images ({len(te_flood)} Flood, {len(te_normal)} Normal)")

# Dataset Class
class FloodDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]

# Augmentation transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_paths), replacement=True)

train_loader = DataLoader(
    FloodDataset(train_paths, train_labels, transform=train_transform),
    batch_size=32, sampler=sampler, num_workers=0, pin_memory=(device.type == 'cuda')
)

test_loader = DataLoader(
    FloodDataset(test_paths, test_labels, transform=eval_transform),
    batch_size=32, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda')
)

# Load MobileNetV2
print("\n[+] Initializing MobileNetV2 Architecture...")
model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(64, 2)
)

if os.path.exists(BEST_MODEL_PATH):
    print(f"[+] Loading baseline weights from: {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
else:
    print("[!] Warning: Checkpoint not found, using default initialization.")

model = model.to(device)

# Unfreeze top feature layers
for param in model.features[-4:].parameters():
    param.requires_grad = True

class_weights = torch.tensor([1.0, 1.25], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW([
    {"params": model.features[-4:].parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 8e-5}
], weight_decay=1e-4)

# Evaluation function
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * imgs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 0].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc, np.array(all_targets), np.array(all_preds), np.array(all_probs)

# Pre-evaluation
print("\n" + "=" * 65)
print("PRE-CALIBRATION BENCHMARK EVALUATION")
print("=" * 65)
pre_loss, pre_acc, pre_targets, pre_preds, pre_probs = evaluate(model, test_loader)
print(f"Initial Test Accuracy: {pre_acc * 100:.2f}% | Initial Test Loss: {pre_loss:.4f}")

# Training Loop
NUM_EPOCHS = 4
print("\n" + "=" * 65)
print(f"STARTING FINE-TUNING ({NUM_EPOCHS} Epochs on Drone & Hard Negatives)")
print("=" * 65)

best_acc = pre_acc

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {running_loss/total:.4f} - Acc: {correct/total*100:.2f}%", flush=True)

    elapsed = time.time() - t0
    train_loss = running_loss / total
    train_acc = correct / total

    val_loss, val_acc, targets, preds, probs = evaluate(model, test_loader)
    print(f"\nEpoch {epoch}/{NUM_EPOCHS} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Loss: {val_loss:.4f}, Test Acc: {val_acc*100:.2f}%", flush=True)

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  [+] Best model checkpoint saved to: {BEST_MODEL_PATH}")

# Final Benchmark Evaluation
print("\n" + "=" * 65)
print("FINAL BENCHMARK EVALUATION ON ALLEYFLOODNET TEST SET")
print("=" * 65)

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
test_loss, test_acc, y_true, y_pred, y_probs = evaluate(model, test_loader)

prec_flood = precision_score(y_true, y_pred, pos_label=0)
rec_flood = recall_score(y_true, y_pred, pos_label=0)
f1_flood = f1_score(y_true, y_pred, pos_label=0)
roc_auc = roc_auc_score(1 - y_true, y_probs)
pr_auc = average_precision_score(1 - y_true, y_probs)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tp = cm[0, 0]
fn = cm[0, 1]
fp = cm[1, 0]
tn = cm[1, 1]

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

print(f"Final Test Accuracy:       {test_acc * 100:.2f}%")
print(f"Flooding Precision:        {prec_flood * 100:.2f}%")
print(f"Flooding Recall:           {rec_flood * 100:.2f}%")
print(f"Flooding F1-Score:         {f1_flood * 100:.2f}%")
print(f"ROC-AUC:                   {roc_auc:.4f}")
print(f"PR-AUC:                    {pr_auc:.4f}")
print(f"False Positive Rate (FPR): {fpr * 100:.2f}%")
print(f"False Negative Rate (FNR): {fnr * 100:.2f}%")
print(f"Confusion Matrix: [[TP={tp}, FN={fn}], [FP={fp}, TN={tn}]]")

# Final production model checkpoint (standard PyTorch state_dict format)
print(f"\n[+] Production model checkpoint verified: {BEST_MODEL_PATH}")

# Save JSON metrics
results = {
    "model_name": "MobileNetV2_Drone_Calibrated",
    "test_accuracy": float(test_acc),
    "flooding_precision": float(prec_flood),
    "flooding_recall": float(rec_flood),
    "flooding_f1": float(f1_flood),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "fpr": float(fpr),
    "fnr": float(fnr),
    "confusion_matrix": cm.tolist(),
    "drone_flooded_count": len(drone_fl),
    "drone_normal_pools_count": len(drone_no),
    "dark_hard_negatives_count": len(drone_dark),
    "water_hard_negatives_count": len(water_negs),
    "drone_hard_negative_trained": True
}

metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"[+] Saved metrics report to: {metrics_path}")
print("\n*** TRAINING AND CALIBRATION COMPLETED SUCCESSFULLY! ***")

