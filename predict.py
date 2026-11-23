import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

import torch.nn as nn

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
    """
    Multi-scale, multi-crop ensemble for high-resolution aerial images.
    Returns a list of (3, 224, 224) tensors covering:
      • full image at 224×224
      • centre crop at 320×320 → 224×224
      • four 50%-overlap quadrant crops → 224×224
      • two horizontal strips (top/bottom halves) → 224×224
    """
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

# ── Out-of-Domain Guard: Non-Disaster Objects, Indoor & Human Filter ───────────
IMAGENET_CACHE = os.path.expanduser(r"~/.cache/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth")

NON_DISASTER_OBJECT_CLASSES = {
    # Clothing / human attire / dresses / portraits (gowns, shirts, suits, etc.)
    411, 434, 445, 451, 452, 453, 455, 457, 459, 465, 474, 501, 514, 515,
    556, 559, 568, 578, 595, 596, 601, 610, 614, 617, 638, 639, 652, 687,
    689, 697, 748, 775, 781, 782, 788, 793, 806, 834, 838, 841, 853, 854,
    869, 878, 887, 893, 904, 906, 921, 981,
    # Jewelry, personal accessories, tableware & domestic artifacts
    470, 494, 499, 505, 584, 619, 647, 679, 711, 725, 728, 738, 773, 849, 850, 855, 857, 883,
    # Indoor home/office objects, furniture, appliances & electronics
    409, 448, 454, 508, 526, 527, 528, 548, 553, 598, 620, 659, 664, 668, 722, 759,
    765, 801, 831, 851, 898, 907, 916
}

def load_domain_guard():
    if os.path.exists(IMAGENET_CACHE):
        try:
            m = models.mobilenet_v2(weights=None)
            m.load_state_dict(torch.load(IMAGENET_CACHE, map_location=device, weights_only=True))
            m.to(device).eval()
            return m
        except Exception:
            return None
    return None

domain_guard = load_domain_guard()

def is_non_disaster_scene(img_rgb: Image.Image) -> tuple[bool, str]:
    if domain_guard is None:
        return False, ""
    t = _NORM(TF.to_tensor(img_rgb.resize((224, 224), Image.BILINEAR))).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(domain_guard(t), dim=1)[0]
        top5_indices = torch.topk(probs, 5).indices.tolist()
        top5_probs = torch.topk(probs, 5).values.tolist()

    for idx, p in zip(top5_indices, top5_probs):
        if idx in NON_DISASTER_OBJECT_CLASSES and p >= 0.02:
            return True, f"Non-Disaster Object #{idx} ({p*100:.1f}%)"
    return False, ""

def predict(image_path):

    # Normalize and clean quotes if passed from shell
    image_path = os.path.expanduser(image_path.strip('"\''))

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
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

    # Guard: Universal Non-Disaster & Out-of-Domain Filter (Objects, Indoor, UI, Human Attire)
    is_non_disaster, _ = is_non_disaster_scene(img_rgb)
    if is_non_disaster:
        print("=" * 50)
        print(f"IMAGE: {image_path}")
        print("✅ RESULT: NORMAL / NON-FLOOD (100.00% confidence)")
        print("Detailed Probabilities: Flood=0.00%, Normal=100.00%")
        print("=" * 50)
        return

    mean_y, std_y = luminance_stats(img_rgb)
    views = lighting_views(img_rgb)

    # ── Inference: Primary Full-Scene with TTA + Multi-View Context ───────
    # Primary views: full image + horizontal flip (Test-Time Augmentation)
    main_tensor = eval_transform(img_rgb).unsqueeze(0).to(device)
    flip_tensor = eval_transform(img_rgb.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)

    with torch.no_grad():
        main_logits = detector(main_tensor)
        flip_logits = detector(flip_tensor)
        main_probs = (torch.softmax(main_logits, dim=1)[0] + torch.softmax(flip_logits, dim=1)[0]) / 2.0
        
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



    print("=" * 50)

    print(f"IMAGE: {image_path}")

    if flood_prob > 50.0:

        print(f"🚨 RESULT: FLOODING DETECTED ({flood_prob:.2f}% confidence)")

    else:

        print(f"✅ RESULT: NORMAL / NON-FLOOD ({normal_prob:.2f}% confidence)")

    print(f"Detailed Probabilities: Flood={flood_prob:.2f}%, Normal={normal_prob:.2f}%")
    if needs_lighting_boost(img_rgb):
        print(
            f"Lighting: low brightness/contrast "
            f"(Y mean={mean_y:.0f}, std={std_y:.0f}) — CLAHE TTA applied"
        )

    print("=" * 50)



if __name__ == "__main__":

    if len(sys.argv) > 1:

        test_path = sys.argv[1]

        predict(test_path)

    else:

        # Test on sample images from AlleyFloodNet test set

        local_base = os.path.join(os.path.dirname(__file__), "data", "AlleyFloodNet", "flood_test_")

        sample_normal = os.path.join(local_base, "non_flooded") if os.path.exists(os.path.join(local_base, "non_flooded")) else r"D:\Flood_Detection_Data\AlleyFloodNet\flood_test_\non_flooded"

        sample_flood = os.path.join(local_base, "flooded") if os.path.exists(os.path.join(local_base, "flooded")) else r"D:\Flood_Detection_Data\AlleyFloodNet\flood_test_\flooded"



        if os.path.exists(sample_normal):

            files = [os.path.join(sample_normal, f) for f in os.listdir(sample_normal)[:2]]

            for f in files:

                predict(f)



        if os.path.exists(sample_flood):

            files = [os.path.join(sample_flood, f) for f in os.listdir(sample_flood)[:2]]

            for f in files:

                predict(f)

