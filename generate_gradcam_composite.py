import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torchvision import transforms, models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "flood_mobilenet_best.pt")

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_images(folder):
    if not os.path.exists(folder): return []
    valid = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid)]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activation = None
        self.hook = target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activation = output
        output.retain_grad()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        output[0, class_idx].backward()
        gradients = self.activation.grad[0].cpu().data.numpy()
        activations = self.activation[0].cpu().data.numpy()
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
        return cam, probs, class_idx

    def close(self):
        self.hook.remove()

def load_detector():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(in_features, 64),
        nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(64, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model

def make_cam_overlay(orig_img, cam, alpha=0.45):
    w, h = orig_img.size
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    cam_norm = np.array(cam_pil) / 255.0
    cmap = matplotlib.colormaps['jet']
    heatmap_rgb = (cmap(cam_norm)[:, :, :3] * 255).astype(np.uint8)
    overlay = Image.blend(orig_img, Image.fromarray(heatmap_rgb), alpha=alpha)
    return cam_norm, overlay

# --- BUILD 6-PANEL COMPOSITE FIGURE ---
model = load_detector()
cam_engine = GradCAM(model, model.features[-1])

flood_imgs = get_images(r"D:\Flood_Detection_Data\AlleyFloodNet\flood_test_\flooded")
normal_imgs = get_images(r"D:\Flood_Detection_Data\AlleyFloodNet\flood_test_\non_flooded")
user_river = r"test_images\Screenshot 2026-10-31 033726.png"

random.seed(42)
sample_flood = random.sample(flood_imgs, 2)
sample_normal = random.sample(normal_imgs, 1)

cases = [
    (sample_flood[0], "True Flood #1"),
    (sample_flood[1], "True Flood #2"),
    (sample_normal[0], "True Normal (Dry)"),
    (user_river, "Natural River (Hard Negative Fixed)"),
]

fig, axes = plt.subplots(len(cases), 3, figsize=(15, 5 * len(cases)))

for row, (img_path, label) in enumerate(cases):
    if not os.path.exists(img_path):
        continue
    with Image.open(img_path) as img:
        orig = img.convert("RGB")

    tensor = eval_transform(orig).unsqueeze(0).to(device)
    cam, probs, pred = cam_engine.generate(tensor)
    flood_pct = probs[0].item() * 100
    norm_pct = probs[1].item() * 100
    is_flood = (pred == 0)
    result_str = "FLOOD" if is_flood else "NORMAL"
    conf = flood_pct if is_flood else norm_pct
    color = "red" if is_flood else "green"

    cam_norm, overlay = make_cam_overlay(orig, cam)

    axes[row, 0].imshow(orig)
    axes[row, 0].set_title(f"Original: {label}", fontweight='bold', fontsize=10)
    axes[row, 0].axis('off')

    axes[row, 1].imshow(cam_norm, cmap='jet')
    axes[row, 1].set_title("Grad-CAM Heatmap", fontweight='bold', fontsize=10)
    axes[row, 1].axis('off')

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f"Prediction: {result_str} ({conf:.1f}%)", fontweight='bold',
                           fontsize=10, color=color)
    axes[row, 2].axis('off')

cam_engine.close()
plt.suptitle("Grad-CAM Explainability — MobileNetV2 Flood Detection\nRed areas = regions that influenced the model decision",
             fontsize=13, fontweight='bold', y=1.005)
plt.tight_layout()

out_path = r"models\gradcam_analysis.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[OK] 4-panel Grad-CAM analysis saved: {out_path}", flush=True)
