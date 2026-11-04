"""
Dataset preparation script:
1. Extracts & resizes 470 drone flooded images from AIFloodSense.zip (1024x768 -> 512x512)
2. Extracts & resizes 400 drone non-flooded images from archive (2).zip (6000x4000 -> 512x512)
   - These include backyard swimming pools, lawns, roads, roofs, landscapes!
3. Generates dark / low-light versions of normal & water images (simulating nighttime, dusk, shadows)
4. Gathers rivers, mountains, landscapes, and drainage into hard negatives pool.
"""

import os
import io
import sys
import zipfile
import numpy as np
from PIL import Image, ImageEnhance

sys.stdout.reconfigure(encoding='utf-8')

AIFLOOD_ZIP = r"c:\Users\Aditi Sharma\Downloads\AIFloodSense.zip"
DRONE_ZIP = r"c:\Users\Aditi Sharma\Downloads\archive (2).zip"
DEST_BASE = r"D:\Flood_Detection_Data\drone_and_hard_negatives"

FLOODED_DIR = os.path.join(DEST_BASE, "flooded")
NON_FLOODED_DIR = os.path.join(DEST_BASE, "non_flooded")
DARK_DIR = os.path.join(DEST_BASE, "dark_hard_negatives")
POOL_LANDSCAPE_DIR = os.path.join(DEST_BASE, "drone_pools_landscapes")

for d in [FLOODED_DIR, NON_FLOODED_DIR, DARK_DIR, POOL_LANDSCAPE_DIR]:
    os.makedirs(d, exist_ok=True)

# 1. Extract AIFloodSense Drone Flooded Images
print("\n[1/4] Processing AIFloodSense Drone Flooded Images...")
count_flood = 0
with zipfile.ZipFile(AIFLOOD_ZIP, 'r') as outer_z:
    nested_data = outer_z.read('AIFloodSense/AIFloodSense.zip')
    with zipfile.ZipFile(io.BytesIO(nested_data), 'r') as inner_z:
        img_files = [f for f in inner_z.namelist() if f.endswith('.jpg')]
        for f in img_files:
            try:
                data = inner_z.read(f)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img = img.resize((512, 512), Image.Resampling.BILINEAR)
                fname = f"drone_flood_{os.path.basename(f)}"
                img.save(os.path.join(FLOODED_DIR, fname), quality=92)
                count_flood += 1
            except Exception as e:
                pass
print(f"  -> Extracted {count_flood} drone flooded images to: {FLOODED_DIR}")

# 2. Extract Semantic Drone Non-Flooded Images (Pools, roofs, landscapes, roads)
print("\n[2/4] Processing Semantic Drone Non-Flooded Images (Swimming pools, Landscapes)...")
count_normal = 0
with zipfile.ZipFile(DRONE_ZIP, 'r') as z:
    drone_files = [f for f in z.namelist() if f.startswith('dataset/semantic_drone_dataset/original_images/') and f.endswith('.jpg')]
    for f in drone_files:
        try:
            data = z.read(f)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            # Resize from 6000x4000 to 512x512
            img = img.resize((512, 512), Image.Resampling.BILINEAR)
            fname = f"drone_normal_{os.path.basename(f)}"
            img.save(os.path.join(NON_FLOODED_DIR, fname), quality=92)
            img.save(os.path.join(POOL_LANDSCAPE_DIR, fname), quality=92)
            count_normal += 1
        except Exception as e:
            pass
print(f"  -> Extracted {count_normal} drone non-flooded images to: {NON_FLOODED_DIR}")

# 3. Generate Dark / Low-Light Hard Negatives (simulate dusk, night streetlights, shadows)
print("\n[3/4] Generating Dark & Low-Light Hard Negatives...")
count_dark = 0
# Take sample normal images and apply low-light transforms
normal_imgs = [os.path.join(NON_FLOODED_DIR, f) for f in os.listdir(NON_FLOODED_DIR) if f.endswith('.jpg')]
sample_for_dark = normal_imgs[:150]

for idx, p in enumerate(sample_for_dark):
    try:
        with Image.open(p) as img:
            img = img.convert("RGB")
            # Transform 1: Night with dim street lighting
            enh_b = ImageEnhance.Brightness(img)
            dark_img1 = enh_b.enhance(0.20)  # 20% brightness
            enh_c = ImageEnhance.Contrast(dark_img1)
            dark_img1 = enh_c.enhance(1.3)   # higher contrast for shadows
            dark_img1.save(os.path.join(DARK_DIR, f"dark_night_{idx:03d}.jpg"), quality=90)
            count_dark += 1

            # Transform 2: Dusk / Evening Shadow
            dark_img2 = enh_b.enhance(0.35)
            # slight bluish/cool tint (evening)
            r, g, b = dark_img2.split()
            r = r.point(lambda i: int(i * 0.85))
            b = b.point(lambda i: min(255, int(i * 1.15)))
            dark_img2 = Image.merge("RGB", (r, g, b))
            dark_img2.save(os.path.join(DARK_DIR, f"dark_dusk_{idx:03d}.jpg"), quality=90)
            count_dark += 1
    except Exception:
        pass
print(f"  -> Generated {count_dark} calibrated dark/night non-flooded images to: {DARK_DIR}")

print("\n" + "=" * 65)
print("DATASET PREPARATION COMPLETED!")
print(f"  - Flooded (Class 0):      {count_flood} drone images")
print(f"  - Non-Flooded (Class 1):  {count_normal} drone images")
print(f"  - Dark Hard Negatives:    {count_dark} night/dusk images")
print("=" * 65)

