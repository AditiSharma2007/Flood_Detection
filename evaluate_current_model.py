"""
Comprehensive model evaluation — all test benchmarks + hard negatives.
Reports: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
Label convention: Class 0 = Flooding, Class 1 = Normal
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

sys.stdout.reconfigure(encoding='utf-8')

BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "flood_mobilenet_best.pt")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(1280, 64),
        nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(64, 2),
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    m.to(DEVICE).eval()
    return m

model = load_model()
print(f"Model loaded: {MODEL_PATH}", flush=True)
print(f"Device: {DEVICE}", flush=True)

# ── Predict helper ────────────────────────────────────────────────────────────
def predict_image(path):
    with Image.open(path) as img:
        t = eval_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    return probs[0].item(), probs[1].item()  # flood_prob, normal_prob

# ── Evaluate a set of (path, true_label) pairs ───────────────────────────────
def evaluate_set(name, items):
    y_true, y_pred, y_flood_prob = [], [], []
    errors = 0
    for path, lbl in items:
        try:
            fp, np_ = predict_image(path)
            y_true.append(lbl)
            y_pred.append(0 if fp > 0.5 else 1)
            y_flood_prob.append(fp)
        except Exception as e:
            errors += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_flood_prob)

    acc  = accuracy_score(y_true, y_pred)
    
    # Flooding = class 0
    prec_f = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_f  = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_f   = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Normal = class 1
    prec_n = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_n  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_n   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # ROC-AUC (flood probability for class 0)
    try:
        auc = roc_auc_score(1 - y_true, y_prob)  # 1-y_true makes flood=1 for AUC
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = cm[0][0]  # flood correctly predicted as flood
    FN = cm[0][1]  # flood missed (predicted normal)
    FP = cm[1][0]  # normal wrongly predicted as flood
    TN = cm[1][1]  # normal correctly predicted as normal

    n_flood  = int((y_true == 0).sum())
    n_normal = int((y_true == 1).sum())

    print(f"\n{'='*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"  Total: {len(y_true)} images (Flood={n_flood}, Normal={n_normal})", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Overall Accuracy:      {acc*100:.2f}%", flush=True)
    print(f"  Flooding Precision:    {prec_f*100:.2f}%", flush=True)
    print(f"  Flooding Recall:       {rec_f*100:.2f}%", flush=True)
    print(f"  Flooding F1-Score:     {f1_f*100:.2f}%", flush=True)
    print(f"  Normal Precision:      {prec_n*100:.2f}%", flush=True)
    print(f"  Normal Recall:         {rec_n*100:.2f}%", flush=True)
    print(f"  Normal F1-Score:       {f1_n*100:.2f}%", flush=True)
    print(f"  ROC-AUC:               {auc:.4f}", flush=True)
    print(f"  Confusion Matrix:  TP={TP} FN={FN} FP={FP} TN={TN}", flush=True)
    print(f"  False Positive Rate:   {FP/(FP+TN)*100:.2f}%" if (FP+TN) > 0 else "", flush=True)
    print(f"  False Negative Rate:   {FN/(TP+FN)*100:.2f}%" if (TP+FN) > 0 else "", flush=True)
    if errors:
        print(f"  Errors loading images: {errors}", flush=True)

    return {
        "name": name, "total": len(y_true), "flood_count": n_flood, "normal_count": n_normal,
        "accuracy": round(acc, 4), "flooding_precision": round(prec_f, 4),
        "flooding_recall": round(rec_f, 4), "flooding_f1": round(f1_f, 4),
        "normal_precision": round(prec_n, 4), "normal_recall": round(rec_n, 4),
        "normal_f1": round(f1_n, 4), "roc_auc": round(auc, 4),
        "confusion_matrix": {"TP": int(TP), "FN": int(FN), "FP": int(FP), "TN": int(TN)},
    }

# ── Collect test sets ─────────────────────────────────────────────────────────
results = {}

# 1. AlleyFloodNet holdout test (262 images)
ALLEY = os.path.join(BASE, "data", "AlleyFloodNet")
alley_items = []
for lbl, folder in [(0, "flooded"), (1, "non_flooded")]:
    d = os.path.join(ALLEY, "flood_test_", folder)
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                alley_items.append((os.path.join(d, f), lbl))

if alley_items:
    results["alleyfloodnet_holdout"] = evaluate_set("AlleyFloodNet Holdout Test (Ground-Level)", alley_items)

# 2. Hard-negative images (should all be Normal)
hn_items = [
    (os.path.join(BASE, "test_images", f), 1)
    for f in ["sunset_lake.jpg", "city_river.jpg", "waterfall_lagoon.jpg",
              "eiffel_tower.jpg", "user_normal_river.jpg", "user_code_screenshot.png",
              "user_flood_1.webp", "user_flood_2.webp"]
    if os.path.exists(os.path.join(BASE, "test_images", f))
]

# Classify test images — flood images are class 0
for i, (p, _) in enumerate(hn_items):
    bn = os.path.basename(p).lower()
    if "flood" in bn:
        hn_items[i] = (p, 0)  # user_flood images are actual floods

if hn_items:
    print(f"\n{'='*60}", flush=True)
    print(f"  User-Provided Test Images (Individual Predictions)", flush=True)
    print(f"{'='*60}", flush=True)
    for path, expected in hn_items:
        fp, np_ = predict_image(path)
        pred = 0 if fp > 0.5 else 1
        pred_str = "🚨 FLOOD" if pred == 0 else "✅ NORMAL"
        exp_str  = "Flood" if expected == 0 else "Normal"
        ok = "✓" if pred == expected else "✗"
        print(f"  {ok} {pred_str} (Flood={fp*100:.1f}%) | expected={exp_str} | {os.path.basename(path)}", flush=True)

# 3. Save results
out_path = os.path.join(BASE, "models", "current_evaluation_metrics.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrics saved to: {out_path}", flush=True)
