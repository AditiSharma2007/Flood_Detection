import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model Path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "flood_mobilenet_best.pt")

# Transforms
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        tensor = eval_transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = detector(tensor)
        probs = torch.softmax(output, dim=1)[0]
        flood_prob = probs[0].item() * 100
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
