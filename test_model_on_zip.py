import os
import sys
import io
import json
import zipfile
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
ZIP_PATH = r"D:\data.zip"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "flood_mobilenet_best.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models", "zip_eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"EVALUATION DEVICE: {device}")
print(f"MODEL CHECKPOINT: {MODEL_PATH}")
print(f"DATA ARCHIVE:     {ZIP_PATH}")
print("=" * 70)

# Preprocessing transforms (identical to training/eval)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
# Extract & Index Dataset from Zip
# -----------------------------------------------------------------------------
def index_zip_dataset(zip_path):
    splits = {"test": [], "valid": [], "train": []}
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if '__MACOSX' in name or name.endswith('/') or '.DS_Store' in name:
                continue
            if not name.lower().endswith(valid_exts):
                continue

            parts = name.split('/')
            # Expected pattern: data/{split}/{class_name}/filename
            if len(parts) >= 4 and parts[0] == 'data':
                split_name = parts[1].lower()
                class_name = parts[2].lower()
                if split_name in splits:
                    # 0: Flooding, 1: Normal
                    if 'flood' in class_name:
                        label = 0
                    elif 'normal' in class_name or 'non' in class_name:
                        label = 1
                    else:
                        continue
                    splits[split_name].append((name, label))

    return splits

# -----------------------------------------------------------------------------
# Evaluation Function
# -----------------------------------------------------------------------------
def evaluate_split(model, zip_path, sample_list, split_name):
    if not sample_list:
        print(f"No samples found for split: {split_name}")
        return None

    y_true = []
    y_pred = []
    y_probs = []
    misclassified = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with zipfile.ZipFile(zip_path, 'r') as z:
        with torch.no_grad():
            for name, target in sample_list:
                data = z.read(name)
                try:
                    with Image.open(io.BytesIO(data)) as img:
                        img_rgb = img.convert("RGB")
                except Exception as e:
                    print(f"Warning: Corrupt image skipped {name}: {e}")
                    continue

                tensor = eval_transform(img_rgb).unsqueeze(0).to(device)
                target_tensor = torch.tensor([target], device=device)

                logits = model(tensor)
                loss = criterion(logits, target_tensor)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=1)[0]
                pred = torch.argmax(probs).item()
                flood_prob = probs[0].item()

                y_true.append(target)
                y_pred.append(pred)
                y_probs.append(flood_prob)

                if pred != target:
                    misclassified.append({
                        "file": name,
                        "true_label": "Flooding" if target == 0 else "Normal",
                        "pred_label": "Flooding" if pred == 0 else "Normal",
                        "flood_confidence": flood_prob,
                        "normal_confidence": probs[1].item()
                    })

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
    avg_loss = total_loss / len(sample_list)

    metrics = {
        "split": split_name,
        "total_samples": len(sample_list),
        "flooded_samples": int(np.sum(y_true == 0)),
        "normal_samples": int(np.sum(y_true == 1)),
        "loss": avg_loss,
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
        "misclassified_count": len(misclassified),
        "misclassified_samples": misclassified
    }

    print("-" * 60)
    print(f"SPLIT: {split_name.upper()} ({len(sample_list)} images: {metrics['flooded_samples']} Flooding, {metrics['normal_samples']} Normal)")
    print(f"  Accuracy:           {acc * 100:.2f}%")
    print(f"  Flooding Precision: {prec_flood * 100:.2f}%")
    print(f"  Flooding Recall:    {rec_flood * 100:.2f}%")
    print(f"  Flooding F1:        {f1_flood * 100:.2f}%")
    print(f"  Normal Recall:      {rec_norm * 100:.2f}%")
    print(f"  ROC-AUC:            {roc_auc:.4f}")
    print(f"  PR-AUC:             {pr_auc:.4f}")
    print(f"  FPR (Normal->Flood):{fpr * 100:.2f}%")
    print(f"  FNR (Flood missed): {fnr * 100:.2f}%")
    print(f"  Confusion Matrix:   [[TP={tp}, FN={fn}], [FP={fp}, TN={tn}]]")
    print("-" * 60)

    return metrics, (y_true, y_pred, y_probs)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_results(all_results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    split_names = [s for s in ["test", "valid", "overall"] if s in all_results]
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
            title=f"{split.upper()} Set\nAcc: {res['accuracy']*100:.1f}% | F1: {res['flooding_f1']*100:.1f}%\nROC-AUC: {res['roc_auc']:.3f}",
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
    plot_path = os.path.join(output_dir, "zip_evaluation_confusion_matrices.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[OK] Saved visualization: {plot_path}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    model = load_model(MODEL_PATH)
    splits = index_zip_dataset(ZIP_PATH)

    all_samples = []
    for s in ["train", "valid", "test"]:
        all_samples.extend(splits[s])

    results = {}
    curves_data = {}

    for split in ["test", "valid", "train"]:
        if splits[split]:
            res, c_data = evaluate_split(model, ZIP_PATH, splits[split], split)
            results[split] = res
            curves_data[split] = c_data

    # Overall dataset
    res_overall, c_overall = evaluate_split(model, ZIP_PATH, all_samples, "overall")
    results["overall"] = res_overall
    curves_data["overall"] = c_overall

    # Save metrics JSON
    json_path = os.path.join(OUTPUT_DIR, "zip_evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[OK] Saved evaluation metrics JSON: {json_path}")

    # Plot
    plot_results(results, OUTPUT_DIR)

if __name__ == "__main__":
    main()

