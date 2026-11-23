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

<<<<<<< HEAD
from PIL import Image

import torchvision.transforms.functional as TF

from torchvision import transforms, models

from lighting_utils import lighting_views, luminance_stats, needs_lighting_boost



# Set UTF-8 encoding

sys.stdout.reconfigure(encoding='utf-8')



# Device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Model Path

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "flood_mobilenet_best.pt")



# ── Transforms ────────────────────────────────────────────────────────────────
# Standard single-scale transform (kept for compatibility)
eval_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

# Normalisation constant (shared across scales)
_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _make_crops(img_rgb: Image.Image):
=======
def predict_single(detector: torch.nn.Module, image_path: str, device: torch.device, threshold: float = 0.5):
>>>>>>> 8a1520b18923c24cccc35a299807dc4db8ed0636
    """
    Runs multi-scale ensemble inference on a single image.
    Returns:
        dict with status, confidence, probabilities, and prediction.
    """
<<<<<<< HEAD
    crops = []
    w, h = img_rgb.size

    def to_tensor(pil_img):
        return _NORM(TF.to_tensor(pil_img.resize((224, 224), Image.BILINEAR)))

    # 1. Full image
    crops.append(to_tensor(img_rgb))

    # 2. Centre crop (inner 70%)
    cx, cy = w // 2, h // 2
    cw, ch = int(w * 0.7), int(h * 0.7)
    crops.append(to_tensor(img_rgb.crop((cx - cw//2, cy - ch//2, cx + cw//2, cy + ch//2))))

    # 3. Four quadrant crops (with 20% overlap each)
    for qr in [0, 1]:
        for qc in [0, 1]:
            x0 = max(0, (qc * w // 2) - w // 10)
            y0 = max(0, (qr * h // 2) - h // 10)
            x1 = min(w, x0 + w // 2 + w // 5)
            y1 = min(h, y0 + h // 2 + h // 5)
            crops.append(to_tensor(img_rgb.crop((x0, y0, x1, y1))))

    # 4. Top and bottom horizontal strips
    crops.append(to_tensor(img_rgb.crop((0, 0,       w, h // 2))))
    crops.append(to_tensor(img_rgb.crop((0, h // 2, w, h))))

    return crops


def load_detector():

    model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(

        nn.Dropout(p=0.4),

        nn.Linear(in_features, 64),

        nn.ReLU(inplace=True),

        nn.Dropout(p=0.2),

        nn.Linear(64, 2)

    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    model.to(device)

    model.eval()

    return model



detector = load_detector()



def predict(image_path):

    # Normalize and clean quotes if passed from shell
    image_path = os.path.expanduser(image_path.strip('"\''))

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
=======
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}
>>>>>>> 8a1520b18923c24cccc35a299807dc4db8ed0636

    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
<<<<<<< HEAD
    except OSError as e:
        if e.errno == 22 or "cloud" in str(e).lower():
            print("=" * 60)
            print(f"❌ Error loading image: {image_path}")
            print("This file is a Microsoft OneDrive 'Files On-Demand' cloud placeholder")
            print("and is not currently downloaded to your local drive.")
            print("\nHow to fix:")
            print("1. Right-click the file in Windows File Explorer and select:")
            print("   'Always keep on this device'")
            print("2. Or make sure OneDrive is running and signed in.")
            print("=" * 60)
        else:
            print(f"❌ Failed to open image ({e.__class__.__name__}): {e}")
        return

    mean_y, std_y = luminance_stats(img_rgb)
    views = lighting_views(img_rgb)

    # ── Inference: Primary Full-Scene + Multi-View Context ─────────────────
    # Primary view: full image at model resolution
    main_tensor = eval_transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        main_logits = detector(main_tensor)
        main_probs = torch.softmax(main_logits, dim=1)[0]
        
        # Optional multi-crop ensemble for scale validation
        crops = []
        for view in views:
            crops.extend(_make_crops(view))
        batch = torch.stack(crops).to(device)
        crop_logits = detector(batch)
        crop_probs = torch.softmax(crop_logits, dim=1).mean(dim=0)
        
        # Weighted decision: 70% primary full scene, 30% ensemble context
        # This prevents zoomed-in crops (e.g. roof tiles) from overriding full flood scenes
        final_probs = 0.70 * main_probs + 0.30 * crop_probs
        
        flood_prob  = final_probs[0].item() * 100
        normal_prob = final_probs[1].item() * 100
=======
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
>>>>>>> 8a1520b18923c24cccc35a299807dc4db8ed0636

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

<<<<<<< HEAD
    print(f"Detailed Probabilities: Flood={flood_prob:.2f}%, Normal={normal_prob:.2f}%")
    if needs_lighting_boost(img_rgb):
        print(
            f"Lighting: low brightness/contrast "
            f"(Y mean={mean_y:.0f}, std={std_y:.0f}) — CLAHE TTA applied"
        )
=======
    args = parser.parse_args()
>>>>>>> 8a1520b18923c24cccc35a299807dc4db8ed0636

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
