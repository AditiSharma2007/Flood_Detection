#!/usr/bin/env python3
"""
Composite Grad-CAM Explainability Figure Generator.
Builds a multi-case visual analysis figure comparing true floods, dry normal scenes,
and difficult hard negatives (e.g. natural river bodies).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.model import load_flood_detector, get_device
from src.transforms import get_eval_transform
from src.gradcam import GradCAM, make_cam_overlay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "flood_mobilenet_best.pt")

def get_images(folder: str) -> list[str]:
    if not os.path.exists(folder):
        return []
    valid = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid)]

def generate_composite():
    device = get_device()
    model = load_flood_detector(checkpoint_path=MODEL_PATH, device=device)
    cam_engine = GradCAM(model, model.features[-1])
    eval_transform = get_eval_transform()

    local_test_base = os.path.join(BASE_DIR, "data", "AlleyFloodNet", "flood_test_")
    flood_dir = os.path.join(local_test_base, "flooded")
    normal_dir = os.path.join(local_test_base, "non_flooded")

    flood_imgs = get_images(flood_dir)
    normal_imgs = get_images(normal_dir)

    user_river = os.path.join(BASE_DIR, "test_images", "Screenshot 2026-10-31 033726.png")

    random.seed(42)
    sample_flood = random.sample(flood_imgs, min(2, len(flood_imgs))) if flood_imgs else []
    sample_normal = random.sample(normal_imgs, min(1, len(normal_imgs))) if normal_imgs else []

    cases = []
    if len(sample_flood) >= 1:
        cases.append((sample_flood[0], "True Flood #1"))
    if len(sample_flood) >= 2:
        cases.append((sample_flood[1], "True Flood #2"))
    if len(sample_normal) >= 1:
        cases.append((sample_normal[0], "True Normal (Dry)"))
    if os.path.exists(user_river):
        cases.append((user_river, "Natural Water (Hard Negative)"))

    if not cases:
        print("No sample images found to generate composite.")
        return

    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 5 * len(cases)), dpi=150)
    if len(cases) == 1:
        axes = np.expand_dims(axes, 0)

    for row, (img_path, label) in enumerate(cases):
        with Image.open(img_path) as img:
            orig = img.convert("RGB")

        tensor = eval_transform(orig).unsqueeze(0).to(device)
        cam, probs, pred = cam_engine.generate(tensor)
        flood_pct = probs[0].item() * 100.0
        norm_pct = probs[1].item() * 100.0
        is_flood = (pred == 0)
        result_str = "FLOOD" if is_flood else "NORMAL"
        conf = flood_pct if is_flood else norm_pct
        color = "red" if is_flood else "green"

        cam_norm, overlay = make_cam_overlay(orig, cam)

        axes[row, 0].imshow(orig)
        axes[row, 0].set_title(f"Original: {label}", fontweight='bold', fontsize=11)
        axes[row, 0].axis('off')

        cax = axes[row, 1].imshow(cam_norm, cmap='jet')
        axes[row, 1].set_title("Grad-CAM Heatmap", fontweight='bold', fontsize=11)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"Prediction: {result_str} ({conf:.1f}%)", fontweight='bold', fontsize=11, color=color)
        axes[row, 2].axis('off')

    cam_engine.close()
    plt.suptitle("Grad-CAM Explainability — MobileNetV2 Flood Detection\nRed areas = regions that influenced the model decision",
                 fontsize=14, fontweight='bold', y=1.002)
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, "models", "gradcam_analysis.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Multi-panel Grad-CAM analysis saved to: {out_path}", flush=True)

if __name__ == "__main__":
    generate_composite()
