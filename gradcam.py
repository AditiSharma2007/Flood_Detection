import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless rendering for maximum speed
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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.hook = self.target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activation = output
        output.retain_grad()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[0, class_idx]
        score.backward()

        gradients = self.activation.grad[0].cpu().data.numpy() # (1280, 7, 7)
        activations = self.activation[0].cpu().data.numpy()     # (1280, 7, 7)

        # Global average pooling on gradients -> channel weights
        weights = np.mean(gradients, axis=(1, 2)) # (1280,)

        # Weighted combination of feature activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32) # (7, 7)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU: keep only features with positive contribution
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
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
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2)
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def explain_prediction(image_path, output_dir="gradcam_outputs"):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}", flush=True)
        return None

    os.makedirs(output_dir, exist_ok=True)
    model = load_detector()
    target_layer = model.features[-1]
    cam_engine = GradCAM(model, target_layer)

    with Image.open(image_path) as img:
        orig_img = img.convert("RGB")
        w, h = orig_img.size

    input_tensor = eval_transform(orig_img).unsqueeze(0).to(device)
    cam, probs, pred_class = cam_engine.generate(input_tensor)
    cam_engine.close()

    flood_prob = probs[0].item() * 100
    normal_prob = probs[1].item() * 100
    is_flood = (pred_class == 0)

    # Resize CAM to original image dimensions
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    cam_norm = np.array(cam_pil) / 255.0

    # Apply JET colormap using matplotlib
    cmap = matplotlib.colormaps['jet']
    heatmap_rgba = cmap(cam_norm)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Blend original image and heatmap (45% opacity)
    overlay_pil = Image.blend(orig_img, heatmap_pil, alpha=0.45)

    # Render publication-quality 3-panel figure
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

    plt.suptitle(f"Grad-CAM Explainability | Predicted: {status_str} (Flood: {flood_prob:.1f}%, Normal: {normal_prob:.1f}%)",
                 fontsize=13, fontweight='bold', y=0.98)
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        explain_prediction(sys.argv[1])
    else:
        sample_img = r"test_images\Screenshot 2026-10-31 033726.png"
        explain_prediction(sample_img)
