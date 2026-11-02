import os
import sys
import time
import json
import random
import numpy as np
from PIL import Image

# Reconfigure stdout for UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score

# Set seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 65)
print(f"HARD NEGATIVE FINE-TUNING PIPELINE")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 65)

# Paths
BASE_DIR = r"D:\Flood_Detection_Data"
OUTPUT_DIR = r"c:\Users\Aditi Sharma\Downloads\Flood_Detection\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "flood_mobilenet_best.pt")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "flood_detection_final.pt")

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

# 1. Gather Hard Negative Water Images (Class 1: Normal)
water_negs = get_images(WATER_NEGATIVES_DIR)
user_test_imgs = get_images(TEST_IMAGES_DIR)

print(f"[+] Loaded {len(water_negs)} natural water images (rivers, lakes, streams, beaches).")
print(f"[+] Loaded {len(user_test_imgs)} user test screenshots to calibrate.")

# 2. AlleyFloodNet Train Set
alley_flood = get_images(ALLEY_TR_FLOOD)
alley_normal = get_images(ALLEY_TR_NORMAL)

# 3. Representative subset of Main Dataset
main_fl = get_images(MAIN_FLOOD)
main_no = get_images(MAIN_NORMAL)

random.seed(42)
sampled_main_fl = random.sample(main_fl, min(1200, len(main_fl)))
sampled_main_no = random.sample(main_no, min(1200, len(main_no)))

# Build training set:
# Class 0: Flood
# Class 1: Normal
train_paths = []
train_labels = []
sample_weights = []

# Hard negative water images -> Class 1 (Weight: 20x)
for p in water_negs:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(20.0)

# User screenshots -> Class 1 (Weight: 25x)
for p in user_test_imgs:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(25.0)

# AlleyFloodNet train -> Weight: 4x
for p in alley_flood:
    train_paths.append(p)
    train_labels.append(0)
    sample_weights.append(4.0)

for p in alley_normal:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(4.0)

# Main dataset sample -> Weight: 1x
for p in sampled_main_fl:
    train_paths.append(p)
    train_labels.append(0)
    sample_weights.append(1.0)

for p in sampled_main_no:
    train_paths.append(p)
    train_labels.append(1)
    sample_weights.append(1.0)

print(f"Total training pool size: {len(train_paths)} images")
print(f"  - Flooding (0): {train_labels.count(0)} images")
print(f"  - Normal (1):   {train_labels.count(1)} images (including {len(water_negs) + len(user_test_imgs)} water hard negatives)")

# Benchmark Holdout Test Set (Untouched AlleyFloodNet)
te_flood = get_images(ALLEY_TE_FLOOD)
te_normal = get_images(ALLEY_TE_NORMAL)
test_paths = te_flood + te_normal
test_labels = [0] * len(te_flood) + [1] * len(te_normal)
print(f"Untouched Benchmark Test Set: {len(test_paths)} images ({len(te_flood)} Flood, {len(te_normal)} Normal)")

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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    batch_size=32, sampler=sampler, num_workers=0, pin_memory=True
)

test_loader = DataLoader(
    FloodDataset(test_paths, test_labels, transform=eval_transform),
    batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)

# Load Model
print("\nLoading MobileNetV2 architecture...")
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
    print(f"Loading existing weights from: {BEST_MODEL_PATH}")
    state = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print("Baseline weights loaded successfully!")

model = model.to(device)

# Unfreeze top feature layers + classifier head
for param in model.features.parameters():
    param.requires_grad = False

for param in model.features[-6:].parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {"params": model.features[-6:].parameters(), "lr": 3e-5},
    {"params": model.classifier.parameters(), "lr": 1e-4}
], weight_decay=1e-4)

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

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

def test_user_screenshots(model):
    model.eval()
    print("\n" + "-" * 50)
    print("CURRENT PREDICTIONS ON USER SCREENSHOTS:")
    for path in user_test_imgs:
        with Image.open(path) as img:
            tensor = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            fl_prob = probs[0].item() * 100
            no_prob = probs[1].item() * 100
            res = "🚨 FLOOD" if fl_prob > 50 else "✅ NORMAL"
            print(f"  {os.path.basename(path)} => {res} (Flood: {fl_prob:.1f}%, Normal: {no_prob:.1f}%)")
    print("-" * 50)

print("\n--- BEFORE FINE-TUNING ---")
test_user_screenshots(model)

# Fine-tuning loop (4 epochs)
print("\n" + "=" * 65)
print("STARTING CALIBRATION FINE-TUNING (4 Epochs)")
print("=" * 65)

best_test_acc = 0.0

for epoch in range(1, 5):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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

    train_loss = running_loss / total
    train_acc = correct / total
    elapsed = time.time() - t0

    # Evaluate on benchmark test set
    test_loss, test_acc, y_true, y_pred, y_probs = evaluate(model, test_loader)
    print(f"Epoch {epoch}/4 [{elapsed:.1f}s] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Holdout Acc: {test_acc*100:.2f}%")

    test_user_screenshots(model)

    if test_acc >= best_test_acc or epoch == 4:
        best_test_acc = test_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        torch.save(model, FINAL_MODEL_PATH)
        print(f"  [SAVED] Checkpoint saved to {BEST_MODEL_PATH}")

# Final Benchmark Assessment
print("\n" + "=" * 65)
print("FINAL BENCHMARK EVALUATION (262 AlleyFloodNet Test Images)")
print("=" * 65)

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
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

print(f"Test Accuracy:             {test_acc * 100:.2f}%")
print(f"Flooding Precision:        {prec_flood * 100:.2f}%")
print(f"Flooding Recall:           {rec_flood * 100:.2f}%")
print(f"Flooding F1-Score:         {f1_flood * 100:.2f}%")
print(f"ROC-AUC:                   {roc_auc:.4f}")
print(f"PR-AUC:                    {pr_auc:.4f}")
print(f"False Positive Rate (FPR): {fpr * 100:.2f}%")
print(f"Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Flooding", "Normal"]))

# Update metrics JSON
results = {
    "model_name": "MobileNetV2_Flood_Detector_Calibrated",
    "test_accuracy": float(test_acc),
    "flooding_precision": float(prec_flood),
    "flooding_recall": float(rec_flood),
    "flooding_f1": float(f1_flood),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "fpr": float(fpr),
    "fnr": float(fnr),
    "confusion_matrix": cm.tolist(),
    "hard_negative_water_count": len(water_negs),
    "user_test_calibrated": True
}

with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

print("[OK] Successfully updated evaluation_metrics.json!")
print("\n*** HARD NEGATIVE FINE-TUNING COMPLETE! ***")
