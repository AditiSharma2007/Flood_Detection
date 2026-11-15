"""Lighting helpers for low-brightness / low-contrast flood images.

Used at training time (synthetic dark/haze/low-contrast augs) and at
inference time (CLAHE + gamma TTA when an image is poorly lit).
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# ImageNet normalisation used by every training / eval script
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Heuristic thresholds on 8-bit luminance (Y)
LOW_BRIGHTNESS_MEAN = 90.0
LOW_CONTRAST_STD = 38.0


def luminance_stats(img: Image.Image) -> tuple[float, float]:
    """Return (mean, std) of Rec.601 luminance in [0, 255]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    y = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return float(y.mean()), float(y.std())


def needs_lighting_boost(img: Image.Image) -> bool:
    mean, std = luminance_stats(img)
    return mean < LOW_BRIGHTNESS_MEAN or std < LOW_CONTRAST_STD


def enhance_lighting(img: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """Lift dark / flat images with CLAHE on L in LAB, then optional gamma."""
    rgb = img.convert("RGB")
    if _HAS_CV2:
        arr = np.asarray(rgb)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        restored = cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2RGB)
        rgb = Image.fromarray(restored)
    else:
        rgb = ImageOps.autocontrast(rgb, cutoff=1)

    mean, _ = luminance_stats(rgb)
    if mean < 80.0:
        # Mid-grey target ~ 110. Clamp gamma so we never bleach the image.
        target = 110.0 / 255.0
        src = max(mean / 255.0, 1e-3)
        gamma = float(np.clip(np.log(target) / np.log(src), 0.55, 1.65))
        lut = [min(255, int(round(255.0 * ((i / 255.0) ** (1.0 / gamma))))) for i in range(256)]
        rgb = rgb.point(lut * 3)
    return rgb


def lighting_views(img: Image.Image) -> list[Image.Image]:
    """Original plus a lighting-normalised copy when the scene is poorly lit."""
    views = [img.convert("RGB")]
    if needs_lighting_boost(views[0]):
        views.append(enhance_lighting(views[0]))
    return views


class RandomLowLightContrast:
    """On-the-fly dusk, underexposure, haze, and washed-out contrast."""

    def __init__(self, p: float = 0.55):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        mode = random.choice(("dark", "low_contrast", "both", "haze", "underexpose"))
        out = img.convert("RGB")

        if mode in ("dark", "both", "underexpose"):
            factor = random.uniform(0.28, 0.78)
            out = ImageEnhance.Brightness(out).enhance(factor)
            if random.random() < 0.5:
                gamma = random.uniform(1.25, 2.1)
                lut = [min(255, int(round(255.0 * ((i / 255.0) ** gamma)))) for i in range(256)]
                out = out.point(lut * 3)

        if mode in ("low_contrast", "both", "haze"):
            out = ImageEnhance.Contrast(out).enhance(random.uniform(0.28, 0.75))

        if mode == "haze":
            # Veiling luminance: mix toward a grey/white sheet (fog, rain, glare).
            veil = Image.new("RGB", out.size, (random.randint(140, 200),) * 3)
            out = Image.blend(out, veil, random.uniform(0.12, 0.38))
            out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

        if random.random() < 0.25:
            # Sensor noise that often accompanies night / phone low-light shots.
            arr = np.asarray(out).astype(np.int16)
            noise = np.random.normal(0, random.uniform(3, 10), arr.shape)
            out = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

        return out
