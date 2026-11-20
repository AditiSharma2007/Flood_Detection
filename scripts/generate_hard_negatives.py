#!/usr/bin/env python3
"""
Generates diverse open-world hard negatives (Class 1: Non-Flood):
- Wooden tables, desks, office furniture, countertops
- Clear skies, cloudy skies, twilight and day gradients
- Indoor walls, tiled floors, ceilings, rooms
- Outdoor green lawns, dry gravel, trees, everyday objects
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_sky_image(w=448, h=448, idx=0):
    # Colors: morning, noon, twilight, overcast
    schemes = [
        ((30, 100, 220), (160, 210, 255)),  # Deep blue to sky blue
        ((70, 130, 180), (200, 225, 250)),  # Steel blue to pale
        ((25, 80, 180),  (135, 206, 250)),  # Vivid sky
        ((220, 130, 80), (255, 210, 150)),  # Twilight / sunset
        ((160, 175, 190),(210, 220, 230)),  # Overcast pale gray/blue
        ((10, 60, 160),  (120, 190, 240)),  # High contrast day
    ]
    top_c, bot_c = schemes[idx % len(schemes)]
    
    # Vertical gradient
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        ratio = y / h
        # Slight curve for atmospheric perspective
        r = top_c[0] * (1 - ratio) + bot_c[0] * ratio
        g = top_c[1] * (1 - ratio) + bot_c[1] * ratio
        b = top_c[2] * (1 - ratio) + bot_c[2] * ratio
        arr[y, :, 0] = r
        arr[y, :, 1] = g
        arr[y, :, 2] = b

    # Optional clouds
    if idx % 3 != 0:
        noise = np.random.normal(0, 15, (h // 4, w // 4)).astype(np.float32)
        noise_img = Image.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8))
        noise_img = noise_img.resize((w, h), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(12))
        c_arr = np.array(noise_img).astype(np.float32)
        mask = (c_arr > 140).astype(np.float32) * ((c_arr - 140) / 115.0)
        for c in range(3):
            arr[:, :, c] = arr[:, :, c] * (1.0 - 0.7 * mask) + 245 * (0.7 * mask)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def create_table_image(w=448, h=448, idx=0):
    # Wood tones: oak, mahogany, walnut, pine, modern laminate
    tones = [
        (139, 69, 19),   # Saddle brown
        (160, 82, 45),   # Sienna
        (205, 133, 63),  # Peru
        (222, 184, 135), # Burlywood
        (101, 67, 33),   # Dark walnut
        (235, 235, 240), # White office laminate
        (60, 60, 65),    # Charcoal desk
        (180, 140, 100), # Light oak
    ]
    base_color = tones[idx % len(tones)]
    
    # Generate wood grain or surface pattern
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = base_color[c]

    # Horizontal wood grain stripes
    grain_freq = random.uniform(0.05, 0.25)
    for y in range(h):
        jitter = np.sin(y * grain_freq) * random.uniform(8, 20)
        arr[y, :, :] += jitter

    # Slight fine noise for wood pores / surface texture
    noise = np.random.normal(0, 5, (h, w, 3))
    arr += noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Add tabletop objects (cup, laptop, notebook, pen)
    draw = ImageDraw.Draw(img)
    if idx % 2 == 0:
        # Mug / Cup
        cx, cy = random.randint(100, w - 100), random.randint(100, h - 100)
        r = random.randint(25, 45)
        cup_color = random.choice([(240, 240, 245), (30, 30, 35), (200, 50, 50), (40, 120, 200)])
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cup_color, outline=(20, 20, 20), width=2)
        draw.ellipse([cx - r + 6, cy - r + 6, cx + r - 6, cy + r - 6], fill=(50, 30, 20)) # coffee
    elif idx % 3 == 0:
        # Laptop / Notebook
        lx, ly = random.randint(60, 200), random.randint(60, 200)
        lw, lh = random.randint(140, 220), random.randint(100, 160)
        draw.rectangle([lx, ly, lx + lw, ly + lh], fill=(40, 45, 50), outline=(200, 200, 200), width=2)
        draw.rectangle([lx + 10, ly + 10, lx + lw - 10, ly + lh - 10], fill=(20, 25, 30))

    return img

def create_indoor_room_image(w=448, h=448, idx=0):
    # Wall colors: beige, light gray, soft green, pale yellow
    wall_colors = [(240, 235, 225), (220, 225, 230), (215, 230, 215), (250, 245, 230), (200, 205, 210)]
    floor_colors = [(180, 130, 90), (120, 80, 50), (80, 80, 85), (210, 205, 195)]

    wc = wall_colors[idx % len(wall_colors)]
    fc = floor_colors[idx % len(floor_colors)]

    img = Image.new("RGB", (w, h), wc)
    draw = ImageDraw.Draw(img)

    # Wall-floor horizon line
    floor_y = int(h * random.uniform(0.55, 0.75))
    draw.rectangle([0, floor_y, w, h], fill=fc)
    # Baseboard
    draw.rectangle([0, floor_y - 8, w, floor_y], fill=(250, 250, 250), outline=(150, 150, 150))

    # Add wall art or door frame
    if idx % 2 == 0:
        fx = random.randint(50, w - 180)
        fy = random.randint(40, floor_y - 140)
        draw.rectangle([fx, fy, fx + 120, fy + 90], fill=(80, 120, 160), outline=(50, 40, 30), width=4)
    else:
        # Window
        wx = random.randint(60, w - 160)
        wy = random.randint(40, floor_y - 160)
        draw.rectangle([wx, wy, wx + 110, wy + 110], fill=(180, 220, 255), outline=(255, 255, 255), width=6)
        draw.line([wx + 55, wy, wx + 55, wy + 110], fill=(255, 255, 255), width=4)
        draw.line([wx, wy + 55, wx + 110, wy + 55], fill=(255, 255, 255), width=4)

    return img

def create_nature_ground_image(w=448, h=448, idx=0):
    # Green grass, autumn leaves, dry soil, gravel
    types = ["grass", "soil", "leaves", "gravel"]
    t = types[idx % len(types)]

    arr = np.zeros((h, w, 3), dtype=np.float32)
    if t == "grass":
        arr[:, :, 0] = np.random.uniform(35, 75, (h, w))
        arr[:, :, 1] = np.random.uniform(110, 175, (h, w))
        arr[:, :, 2] = np.random.uniform(25, 55, (h, w))
    elif t == "soil":
        arr[:, :, 0] = np.random.uniform(120, 160, (h, w))
        arr[:, :, 1] = np.random.uniform(80, 115, (h, w))
        arr[:, :, 2] = np.random.uniform(40, 70, (h, w))
    elif t == "leaves":
        arr[:, :, 0] = np.random.uniform(160, 210, (h, w))
        arr[:, :, 1] = np.random.uniform(90, 140, (h, w))
        arr[:, :, 2] = np.random.uniform(20, 60, (h, w))
    else:
        # Gravel
        gray = np.random.uniform(90, 170, (h, w))
        for c in range(3):
            arr[:, :, c] = gray

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(1.2))

def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "open_world_negatives", "non_flooded")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating diverse open-world hard negatives (Class 1: Non-Flood)...")
    count = 0

    # 1. Skies (100)
    for i in range(100):
        img = create_sky_image(idx=i)
        img.save(os.path.join(out_dir, f"neg_sky_{i:03d}.jpg"), quality=92)
        count += 1

    # 2. Tables & Desks (100)
    for i in range(100):
        img = create_table_image(idx=i)
        img.save(os.path.join(out_dir, f"neg_table_{i:03d}.jpg"), quality=92)
        count += 1

    # 3. Indoor Rooms & Walls (100)
    for i in range(100):
        img = create_indoor_room_image(idx=i)
        img.save(os.path.join(out_dir, f"neg_indoor_{i:03d}.jpg"), quality=92)
        count += 1

    # 4. Nature & Ground textures (100)
    for i in range(100):
        img = create_nature_ground_image(idx=i)
        img.save(os.path.join(out_dir, f"neg_nature_{i:03d}.jpg"), quality=92)
        count += 1

    print(f"Successfully generated {count} open-world hard negatives in: {out_dir}")

if __name__ == "__main__":
    main()
