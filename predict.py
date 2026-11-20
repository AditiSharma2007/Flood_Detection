#!/usr/bin/env python3
"""
Flood Detection: Production Inference Engine.
Performs multi-scale crop ensemble inference on single images or entire directories.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
from PIL import Image
import torch

from src.model import load_flood_detector, get_device
from src.transforms import make_crops

def predict_single(detector: torch.nn.Module, image_path: str, device: torch.device, threshold: float = 0.5):
    """
    Runs multi-scale ensemble inference on a single image.
    Returns:
        dict with status, confidence, probabilities, and prediction.
    """
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
    except Exception as e:
        return {"error": f"Failed to load image {image_path}: {e}"}

    crops = make_crops(img_rgb)
    batch = torch.stack(crops).to(device)

    with torch.no_grad():
        logits = detector(batch)
        probs_all = torch.softmax(logits, dim=1)
        probs = probs_all.mean(dim=0)

        flood_prob = probs[0].item() * 100.0
        normal_prob = probs[1].item() * 100.0

    is_flood = (flood_prob / 100.0) >= threshold
    status = "FLOODING DETECTED" if is_flood else "NORMAL / NON-FLOOD"
    confidence = flood_prob if is_flood else normal_prob

    return {
        "image_path": image_path,
        "prediction": "Flooding" if is_flood else "Normal",
        "status": status,
        "confidence_pct": round(confidence, 2),
        "flood_prob_pct": round(flood_prob, 2),
        "normal_prob_pct": round(normal_prob, 2),
        "is_flood": is_flood
    }

def print_result(res: dict):
    if "error" in res:
        print(f"❌ ERROR: {res['error']}")
        return

    print("=" * 55)
    print(f"IMAGE: {res['image_path']}")
    if res["is_flood"]:
        print(f"🚨 RESULT: {res['status']} ({res['confidence_pct']:.2f}% confidence)")
    else:
        print(f"✅ RESULT: {res['status']} ({res['confidence_pct']:.2f}% confidence)")
    print(f"Detailed Probabilities: Flood={res['flood_prob_pct']:.2f}%, Normal={res['normal_prob_pct']:.2f}%")
    print("=" * 55)

def main():
    parser = argparse.ArgumentParser(description="Flood Detection Multi-Crop Inference")
    parser.add_argument("input_path", nargs="?", default=None, help="Path to image file or directory of images")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold for flooding (default: 0.5)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Inference device")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--model-path", type=str, default=None, help="Custom model checkpoint path")

    args = parser.parse_args()

    device = get_device(args.device)
    try:
        detector = load_flood_detector(checkpoint_path=args.model_path, device=device)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # If no input provided, fallback to test samples in local workspace
    if args.input_path is not None:
        args.input_path = args.input_path.strip("'\"")

    if args.input_path is None:
        local_base = os.path.join(os.path.dirname(__file__), "data", "AlleyFloodNet", "flood_test_")
        sample_normal = os.path.join(local_base, "non_flooded")
        sample_flood = os.path.join(local_base, "flooded")
        sample_images = []

        if os.path.exists(sample_normal):
            sample_images.extend([os.path.join(sample_normal, f) for f in os.listdir(sample_normal)[:2] if f.lower().endswith(('.jpg', '.png', '.webp'))])
        if os.path.exists(sample_flood):
            sample_images.extend([os.path.join(sample_flood, f) for f in os.listdir(sample_flood)[:2] if f.lower().endswith(('.jpg', '.png', '.webp'))])

        if not sample_images:
            # Check test_images/ folder
            test_dir = os.path.join(os.path.dirname(__file__), "test_images")
            if os.path.exists(test_dir):
                sample_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir)[:4] if f.lower().endswith(('.jpg', '.png', '.webp'))]

        targets = sample_images
    elif os.path.isdir(args.input_path):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        targets = [os.path.join(args.input_path, f) for f in sorted(os.listdir(args.input_path)) if f.lower().endswith(valid_exts)]
    else:
        targets = [args.input_path]

    results = []
    for img_p in targets:
        res = predict_single(detector, img_p, device, threshold=args.threshold)
        results.append(res)
        if not args.json:
            print_result(res)

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))

if __name__ == "__main__":
    main()
