import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

import torch.nn as nn

from PIL import Image

import torchvision.transforms.functional as TF

from torchvision import transforms, models



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



def predict(image_path):

    if not os.path.exists(image_path):

        print(f"File not found: {image_path}")

        return



    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")

    # ── Multi-scale ensemble inference ───────────────────────────────────────
    crops = _make_crops(img_rgb)
    batch = torch.stack(crops).to(device)   # (N, 3, 224, 224)

    with torch.no_grad():
        logits = detector(batch)             # (N, 2)
        # Average softmax probabilities across all crops (max-vote alternative)
        probs_all = torch.softmax(logits, dim=1)  # (N, 2)
        probs = probs_all.mean(dim=0)             # (2,)

        flood_prob  = probs[0].item() * 100
        normal_prob = probs[1].item() * 100



    print("=" * 50)

    print(f"IMAGE: {image_path}")

    if flood_prob > 50.0:

        print(f"🚨 RESULT: FLOODING DETECTED ({flood_prob:.2f}% confidence)")

    else:

        print(f"✅ RESULT: NORMAL / NON-FLOOD ({normal_prob:.2f}% confidence)")

    print(f"Detailed Probabilities: Flood={flood_prob:.2f}%, Normal={normal_prob:.2f}%")

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

