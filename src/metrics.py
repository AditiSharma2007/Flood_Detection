"""
Standardized performance metrics evaluation for Flood Detection.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_flood_probs: np.ndarray) -> dict:
    """
    Computes full benchmark metrics following the project convention:
      - Class 0: Flooding (positive class for detection)
      - Class 1: Normal (negative class)
    """
    acc = accuracy_score(y_true, y_pred)
    prec_flood = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_flood = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_flood = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    prec_norm = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_norm = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_norm = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Note: 1 - y_true maps class 0 (flood) to 1 for standard ROC-AUC / PR-AUC calculation
    try:
        roc_auc = roc_auc_score(1 - y_true, y_flood_probs)
    except Exception:
        roc_auc = float('nan')

    try:
        pr_auc = average_precision_score(1 - y_true, y_flood_probs)
    except Exception:
        pr_auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
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
        "fnr": fnr
    }
