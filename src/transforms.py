"""
Image transforms and multi-crop preprocessing for Flood Detection.
"""

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_eval_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

def get_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

_NORM = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

def make_crops(img_rgb: Image.Image, target_size: int = 224) -> list[torch.Tensor]:
    """
    Multi-scale, multi-crop ensemble for high-resolution aerial and ground images.
    Returns a list of (3, target_size, target_size) tensors covering:
      - full image resized to target_size
      - centre crop (inner 70%)
      - four overlapping quadrant crops (20% overlap)
      - two horizontal strips (top/bottom halves)
    """
    crops = []
    w, h = img_rgb.size

    def to_tensor(pil_img):
        return _NORM(TF.to_tensor(pil_img.resize((target_size, target_size), Image.BILINEAR)))

    # 1. Full image
    crops.append(to_tensor(img_rgb))

    # 2. Centre crop (inner 70%)
    cx, cy = w // 2, h // 2
    cw, ch = int(w * 0.7), int(h * 0.7)
    crops.append(to_tensor(img_rgb.crop((cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2))))

    # 3. Four quadrant crops (with 20% overlap)
    for qr in [0, 1]:
        for qc in [0, 1]:
            x0 = max(0, (qc * w // 2) - w // 10)
            y0 = max(0, (qr * h // 2) - h // 10)
            x1 = min(w, x0 + w // 2 + w // 5)
            y1 = min(h, y0 + h // 2 + h // 5)
            crops.append(to_tensor(img_rgb.crop((x0, y0, x1, y1))))

    # 4. Top and bottom horizontal strips
    crops.append(to_tensor(img_rgb.crop((0, 0, w, h // 2))))
    crops.append(to_tensor(img_rgb.crop((0, h // 2, w, h))))

    return crops
