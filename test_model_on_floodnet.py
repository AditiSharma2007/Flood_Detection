import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLOODNET_DIR = os.path.join(BASE_DIR, "data", "FloodNet", "FloodNet-Supervised_v1.0")
MODEL_PATH = os.path.join(BASE_DIR, "models", "flood_mobilenet_best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "floodnet_eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"FLOODNET BENCHMARK EVALUATION")
print(f"Device:           {device}")
print(f"Model Checkpoint: {MODEL_PATH}")
print(f"FloodNet Data:    {FLOODNET_DIR}")
print("=" * 70)

# Preprocessing transform
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------------------
# Model Loader
# -----------------------------------------------------------------------------
def load_model(path):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2)
    )
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Index Dataset & Extract Ground Truth Labels from Masks
# -----------------------------------------------------------------------------
def index_floodnet_split(split_name):
    org_dir = os.path.join(FLOODNET_DIR, split_name, f"{split_name}-org-img")
    lbl_dir = os.path.join(FLOODNET_DIR, split_name, f"{split_name}-label-img")

    if not os.path.exists(org_dir) or not os.path.exists(lbl_dir):
        print(f"Split directory not found: {split_name}")
        return []

    org_files = sorted([f for f in os.listdir(org_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    samples = []

    print(f"Indexing {split_name} split ({len(org_files)} images)...")
    t0 = time.time()

    for fname in org_files:
        base_id = os.path.splitext(fname)[0]
        mask_name = f"{base_id}_lab.png"
        mask_path = os.path.join(lbl_dir, mask_name)

        if not os.path.exists(mask_path):
            # Fallback check
            mask_name = f"{base_id}.png"
            mask_path = os.path.join(lbl_dir, mask_name)
            if not os.path.exists(mask_path):
                continue

        # Fast color palette inspection
        with Image.open(mask_path) as m_img:
            colors = m_img.getcolors(maxcolors=50)

        px_bld_flood = 0
        px_road_flood = 0
        px_water = 0
        px_pool = 0

        if colors:
            for count, val in colors:
                if val == 1:
                    px_bld_flood += count
                elif val == 3:
                    px_road_flood += count
                elif val == 5:
                    px_water += count
                elif val == 8:
                    px_pool += count

        # Classification ground truth:
        # Label 0 = Flooding, Label 1 = Normal
        has_flood = (px_bld_flood > 0) or (px_road_flood > 0) or (px_water > 0)
        label = 0 if has_flood else 1

        img_path = os.path.join(org_dir, fname)
        samples.append({
            "img_path": img_path,
            "mask_path": mask_path,
            "filename": fname,
            "label": label,
            "px_bld_flood": px_bld_flood,
            "px_road_flood": px_road_flood,
            "px_water": px_water,
            "px_pool": px_pool
        })

    elapsed = time.time() - t0
    n_flood = sum(1 for s in samples if s["label"] == 0)
    n_normal = sum(1 for s in samples if s["label"] == 1)
    print(f"  Indexed {len(samples)} images in {elapsed:.2f}s: {n_flood} Flooded, {n_normal} Normal (Non-Flooded)")
    return samples

# -----------------------------------------------------------------------------
# Evaluation Loop
# -----------------------------------------------------------------------------
def evaluate_samples(model, samples, split_name):
    y_true = []
    y_pred = []
    y_probs = []
    misclassified = []

    # Detailed sub-metrics: pool test
    pool_total = 0
    pool_correct_normal = 0

    # Detailed sub-metrics: flooded infrastructure test
    infra_flood_total = 0
    infra_flood_detected = 0

    t0 = time.time()
    with torch.no_grad():
        for idx, item in enumerate(samples):
            img_path = item["img_path"]
            target = item["label"]

            try:
                with Image.open(img_path) as img:
                    img.draft("RGB", (448, 448))
                    img_rgb = img.convert("RGB")
                    img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

            tensor = eval_transform(img_resized).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = torch.argmax(probs).item()
            flood_prob = probs[0].item()

            y_true.append(target)
            y_pred.append(pred)
            y_probs.append(flood_prob)

            # Check swimming pool hard negatives
            if item["px_pool"] > 0 and target == 1:
                pool_total += 1
                if pred == 1:
                    pool_correct_normal += 1

            # Check infrastructure flood (buildings or roads submerged)
            if (item["px_bld_flood"] > 0 or item["px_road_flood"] > 0):
                infra_flood_total += 1
                if pred == 0:
                    infra_flood_detected += 1

            if pred != target:
                misclassified.append({
                    "filename": item["filename"],
                    "true_label": "Flooding" if target == 0 else "Normal",
                    "pred_label": "Flooding" if pred == 0 else "Normal",
                    "flood_confidence": flood_prob,
                    "normal_confidence": probs[1].item(),
                    "px_bld_flood": item["px_bld_flood"],
                    "px_road_flood": item["px_road_flood"],
                    "px_water": item["px_water"],
                    "px_pool": item["px_pool"]
                })

            if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
                pct = (idx + 1) / len(samples) * 100
                print(f"  Processed {idx + 1}/{len(samples)} images ({pct:.1f}%)...")

    elapsed = time.time() - t0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = accuracy_score(y_true, y_pred)
    prec_flood = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_flood = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_flood = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    prec_norm = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_norm = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_norm = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        roc_auc = roc_auc_score(1 - y_true, y_probs)
    except Exception:
        roc_auc = float('nan')

    try:
        pr_auc = average_precision_score(1 - y_true, y_probs)
    except Exception:
        pr_auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    pool_acc = (pool_correct_normal / pool_total * 100) if pool_total > 0 else 100.0
    infra_recall = (infra_flood_detected / infra_flood_total * 100) if infra_flood_total > 0 else 100.0

    metrics = {
        "split": split_name,
        "total_images": len(samples),
        "flooded_images": int(np.sum(y_true == 0)),
        "normal_images": int(np.sum(y_true == 1)),
        "elapsed_seconds": elapsed,
        "accuracy": acc,
        "flooding_precision": prec_flood,
        "flooding_recall": rec_flood,
        "flooding_f1": f1_flood,
        "normal_precision": prec_norm,
        "normal_recall": rec_norm,
        "normal_f1": f1_norm,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "fpr": fpr,
        "fnr": fnr,
        "sub_metrics": {
            "swimming_pool_scenes": pool_total,
            "swimming_pool_correct_normal": pool_correct_normal,
            "swimming_pool_accuracy": pool_acc,
            "flooded_infrastructure_scenes": infra_flood_total,
            "flooded_infrastructure_detected": infra_flood_detected,
            "flooded_infrastructure_recall": infra_recall
        },
        "misclassified_count": len(misclassified),
        "misclassified_samples": misclassified
    }

    print("-" * 65)
    print(f"SPLIT: {split_name.upper()} RESULTS ({len(samples)} images evaluated in {elapsed:.1f}s)")
    print(f"  Accuracy:                      {acc * 100:.2f}%")
    print(f"  Flooding Recall:               {rec_flood * 100:.2f}% (Floods caught)")
    print(f"  Flooding Precision:            {prec_flood * 100:.2f}%")
    print(f"  Flooding F1-Score:             {f1_flood * 100:.2f}%")
    print(f"  Normal Specificity:            {rec_norm * 100:.2f}% (Normal correctly identified)")
    print(f"  ROC-AUC:                       {roc_auc:.4f}")
    print(f"  PR-AUC:                        {pr_auc:.4f}")
    print(f"  False Positive Rate (FPR):     {fpr * 100:.2f}%")
    print(f"  False Negative Rate (FNR):     {fnr * 100:.2f}%")
    print(f"  Confusion Matrix:              [[TP={tp}, FN={fn}], [FP={fp}, TN={tn}]]")
    print(f"  Swimming Pool Scenes (Hard Neg): {pool_correct_normal}/{pool_total} ({pool_acc:.1f}% correct Normal)")
    print(f"  Submerged Roads/Buildings:     {infra_flood_detected}/{infra_flood_total} ({infra_recall:.1f}% recall)")
    print("-" * 65)

    return metrics, (y_true, y_pred, y_probs)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_results(all_results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    split_names = [s for s in ["test", "val", "overall"] if s in all_results]

    for idx, split in enumerate(split_names):
        res = all_results[split]
        cm_dict = res["confusion_matrix"]
        cm = np.array([[cm_dict["TP"], cm_dict["FN"]], [cm_dict["FP"], cm_dict["TN"]]])
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        classes = ["Flooding", "Normal"]
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=classes,
            yticklabels=classes,
            title=f"FloodNet {split.upper()} Benchmark\nAcc: {res['accuracy']*100:.1f}% | Recall: {res['flooding_recall']*100:.1f}%\nROC-AUC: {res['roc_auc']:.3f}",
            ylabel='Actual Label',
            xlabel='Predicted Label'
        )

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cell_text = f"{cm[i, j]}"
                desc = "(TP)" if (i==0 and j==0) else "(FN)" if (i==0 and j==1) else "(FP)" if (i==1 and j==0) else "(TN)"
                ax.text(j, i, f"{cell_text}\n{desc}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "floodnet_confusion_matrices.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[OK] Saved visualization: {plot_path}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    model = load_model(MODEL_PATH)

    test_samples = index_floodnet_split("test")
    val_samples = index_floodnet_split("val")

    all_samples = test_samples + val_samples

    results = {}
    if test_samples:
        m_test, _ = evaluate_samples(model, test_samples, "test")
        results["test"] = m_test

    if val_samples:
        m_val, _ = evaluate_samples(model, val_samples, "val")
        results["val"] = m_val

    if all_samples:
        m_overall, _ = evaluate_samples(model, all_samples, "overall")
        results["overall"] = m_overall

    json_path = os.path.join(OUTPUT_DIR, "floodnet_evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[OK] Saved FloodNet metrics JSON: {json_path}")

    plot_results(results, OUTPUT_DIR)

if __name__ == "__main__":
    main()

