#!/usr/bin/env python3
"""
Grad-CAM Explainability Visualizer for Flood Detection.
Generates publication-quality 3-panel decision heatmaps.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.model import load_flood_detector, get_device
from src.transforms import get_eval_transform
from src.gradcam import GradCAM, make_cam_overlay

def explain_prediction(image_path: str, output_dir: str = "gradcam_outputs", device: torch.device = None):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}", flush=True)
        return None

    if device is None:
        device = get_device()

    os.makedirs(output_dir, exist_ok=True)
    model = load_flood_detector(device=device)
    target_layer = model.features[-1]
    cam_engine = GradCAM(model, target_layer)
    eval_transform = get_eval_transform()

    with Image.open(image_path) as img:
        orig_img = img.convert("RGB")
        w, h = orig_img.size

    input_tensor = eval_transform(orig_img).unsqueeze(0).to(device)
    cam, probs, pred_class = cam_engine.generate(input_tensor)
    cam_engine.close()

    flood_prob = probs[0].item() * 100.0
    normal_prob = probs[1].item() * 100.0
    is_flood = (pred_class == 0)

    cam_norm, overlay_pil = make_cam_overlay(orig_img, cam, alpha=0.45)

    # Render 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_img)
    axes[0].set_title("1. Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    cax = axes[1].imshow(cam_norm, cmap='jet')
    axes[1].set_title("2. Grad-CAM Activation Heatmap", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(cax, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay_pil)
    status_str = "FLOODING DETECTED" if is_flood else "NORMAL / NON-FLOOD"
    confidence = flood_prob if is_flood else normal_prob
    color_tag = "red" if is_flood else "green"
    axes[2].set_title(f"3. Decision Overlay ({confidence:.1f}% Conf)", fontsize=12, fontweight='bold', color=color_tag)
    axes[2].axis('off')

    plt.suptitle(
        f"Grad-CAM Explainability | Predicted: {status_str} (Flood: {flood_prob:.1f}%, Normal: {normal_prob:.1f}%)",
        fontsize=13, fontweight='bold', y=0.98
    )
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_filename = f"{base_name}_gradcam.png"
    out_path = os.path.join(output_dir, out_filename)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

    print("=" * 60, flush=True)
    print(f"IMAGE: {image_path}", flush=True)
    if is_flood:
        print(f"🚨 RESULT: FLOODING DETECTED ({flood_prob:.2f}% confidence)", flush=True)
    else:
        print(f"✅ RESULT: NORMAL / NON-FLOOD ({normal_prob:.2f}% confidence)", flush=True)
    print(f"Detailed Probabilities: Flood={flood_prob:.2f}%, Normal={normal_prob:.2f}%", flush=True)
    print(f"🔍 Grad-CAM Visual Explanation Saved: {out_path}", flush=True)
    print("=" * 60, flush=True)

    return out_path

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Explainability Visualizer")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="gradcam_outputs", help="Output directory for heatmaps")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Inference device")

    args = parser.parse_args()

    target_img = args.image_path
    if target_img is not None:
        target_img = target_img.strip("'\"")
    if target_img is None:
        sample_candidates = [
            os.path.join("test_images", "Screenshot 2026-10-31 033726.png"),
            os.path.join("test_images", "user_flood_1.webp"),
            os.path.join("test_images", "user_flood_2.webp")
        ]
        for c in sample_candidates:
            if os.path.exists(c):
                target_img = c
                break

    if target_img and os.path.exists(target_img):
        device = get_device(args.device)
        explain_prediction(target_img, output_dir=args.output_dir, device=device)
    else:
        print("No valid test image found. Please provide an image path.")

if __name__ == "__main__":
    main()
