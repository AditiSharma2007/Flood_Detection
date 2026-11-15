"""Live evaluation of the current checkpoint following the 6-stage pipeline."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torchvision import models, transforms

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "flood_mobilenet_best.pt"
ALLEY_TEST = BASE / "data" / "AlleyFloodNet" / "flood_test_"
OUT_JSON = BASE / "models" / "live_architecture_eval.json"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eval_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def collect_labeled(folder, label):
    items = []
    if not folder.exists():
        return items
    for f in folder.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            items.append((str(f), label))
    return items


def predict_probs(model, paths):
    flood_scores = []
    preds = []
    for path in paths:
        with Image.open(path) as img:
            tensor = eval_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        flood_p = float(probs[0].item())
        flood_scores.append(flood_p)
        preds.append(0 if flood_p > 0.5 else 1)
    return np.array(flood_scores), np.array(preds)


def metrics_from(y_true, y_pred, flood_scores):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # rows: true 0/1, cols: pred 0/1  -> TP floods = cm[0,0]
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])
    flood_pos_scores = 1.0 - flood_scores  # unused; ROC uses flood class score
    out = {
        "n": int(len(y_true)),
        "n_flood": int((y_true == 0).sum()),
        "n_normal": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "flooding_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "flooding_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "flooding_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "normal_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "normal_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fpr": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "fnr": float(fn / (fn + tp) if (fn + tp) else 0.0),
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Flood", "Non-Flood"], digits=4, zero_division=0
        ),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true == 0, flood_scores))
        out["pr_auc"] = float(average_precision_score(y_true == 0, flood_scores))
    return out


def main():
    print("=" * 70)
    print("STAGE 5 LIVE EVAL — MobileNetV2 flood classifier")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {MODEL_PATH}")
    print("=" * 70)

    model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Classifier head:\n{model.classifier}")

    flood_items = collect_labeled(ALLEY_TEST / "flooded", 0)
    normal_items = collect_labeled(ALLEY_TEST / "non_flooded", 1)
    alley = flood_items + normal_items
    print(f"\nAlleyFloodNet holdout: flood={len(flood_items)}  non-flood={len(normal_items)}")

    alley_paths = [p for p, _ in alley]
    alley_y = [y for _, y in alley]
    flood_scores, preds = predict_probs(model, alley_paths)
    alley_metrics = metrics_from(alley_y, preds, flood_scores)

    print("\n--- AlleyFloodNet test (stage 1+2+5 inference: resize 224, ImageNet norm) ---")
    print(f"Accuracy:           {alley_metrics['accuracy']*100:.2f}%")
    print(f"Flood precision:    {alley_metrics['flooding_precision']*100:.2f}%")
    print(f"Flood recall:       {alley_metrics['flooding_recall']*100:.2f}%")
    print(f"Flood F1:           {alley_metrics['flooding_f1']*100:.2f}%")
    print(f"Non-flood recall:   {alley_metrics['normal_recall']*100:.2f}%")
    print(f"FPR / FNR:          {alley_metrics['fpr']*100:.2f}% / {alley_metrics['fnr']*100:.2f}%")
    print(f"ROC-AUC / PR-AUC:   {alley_metrics['roc_auc']:.4f} / {alley_metrics['pr_auc']:.4f}")
    cm = alley_metrics["confusion_matrix"]
    print(f"Confusion: TP={cm['TP']} FN={cm['FN']} FP={cm['FP']} TN={cm['TN']}")
    print(alley_metrics["classification_report"])

    # Stage 3: hard-negative water / user images
    hard_neg_expected_normal = [
        BASE / "test_images" / "sunset_lake.jpg",
        BASE / "test_images" / "city_river.jpg",
        BASE / "test_images" / "waterfall_lagoon.jpg",
        BASE / "test_images" / "eiffel_tower.jpg",
        BASE / "test_images" / "user_normal_river.jpg",
        BASE / "test_images" / "user_code_screenshot.png",
    ]
    user_flood = [
        BASE / "test_images" / "user_flood_1.webp",
        BASE / "test_images" / "user_flood_2.webp",
    ]
    extra_shots = [
        p for p in (BASE / "test_images").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and p not in hard_neg_expected_normal
        and p not in user_flood
    ]

    print("\n--- Stage 3 hard-negative water (expected Non-Flood) ---")
    hn_rows = []
    hn_ok = 0
    for path in hard_neg_expected_normal:
        if not path.exists():
            print(f"  MISSING {path.name}")
            continue
        scores, pred = predict_probs(model, [str(path)])
        flood_p = scores[0] * 100
        normal_p = 100 - flood_p
        ok = pred[0] == 1
        hn_ok += int(ok)
        icon = "OK" if ok else "FALSE ALARM"
        print(f"  [{icon:12}] Flood={flood_p:6.1f}%  Normal={normal_p:6.1f}%  {path.name}")
        hn_rows.append(
            {
                "file": path.name,
                "expected": "Non-Flood",
                "pred": "Flood" if pred[0] == 0 else "Non-Flood",
                "flood_pct": flood_p,
                "normal_pct": normal_p,
                "correct": bool(ok),
            }
        )

    print("\n--- User flood images (expected Flood) ---")
    uf_rows = []
    for path in user_flood:
        if not path.exists():
            continue
        scores, pred = predict_probs(model, [str(path)])
        flood_p = scores[0] * 100
        ok = pred[0] == 0
        icon = "OK" if ok else "MISS"
        print(f"  [{icon:12}] Flood={flood_p:6.1f}%  Normal={100-flood_p:6.1f}%  {path.name}")
        uf_rows.append(
            {
                "file": path.name,
                "expected": "Flood",
                "pred": "Flood" if pred[0] == 0 else "Non-Flood",
                "flood_pct": flood_p,
                "correct": bool(ok),
            }
        )

    print("\n--- Other test_images ---")
    extra_rows = []
    for path in extra_shots:
        scores, pred = predict_probs(model, [str(path)])
        flood_p = scores[0] * 100
        label = "Flood" if pred[0] == 0 else "Non-Flood"
        print(f"  [{label:10}] Flood={flood_p:6.1f}%  {path.name}")
        extra_rows.append({"file": path.name, "pred": label, "flood_pct": flood_p})

    payload = {
        "checkpoint": str(MODEL_PATH),
        "device": str(DEVICE),
        "preprocessing": "resize 224x224, ImageNet mean/std",
        "backbone": "MobileNetV2 + dual-dropout classifier (1280->64->2)",
        "domain_adaptation": "supervised fine-tune on AlleyFloodNet target streets (not adversarial DA)",
        "alleyfloodnet_holdout": alley_metrics,
        "hard_negative_water": hn_rows,
        "hard_negative_correct": hn_ok,
        "hard_negative_total": len(hn_rows),
        "user_flood": uf_rows,
        "other_test_images": extra_rows,
    }
    # classification_report is a string; keep it
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
