
import numpy as np
import cv2
import math
from .realtex import load_texture_paths, sample_texture

def _dir_kernel(angle_deg: float, k: int):
    # Create a line kernel at angle
    angle = np.deg2rad(angle_deg)
    kern = np.zeros((k, k), np.float32)
    cx = cy = k//2
    # draw a line across kernel
    for t in np.linspace(-1, 1, 2*k):
        x = int(cx + t * (k//2) * np.cos(angle))
        y = int(cy + t * (k//2) * np.sin(angle))
        if 0 <= x < k and 0 <= y < k:
            kern[y, x] = 1.0
    s = kern.sum()
    if s > 0:
        kern /= s
    else:
        kern[cy, :] = 1.0 / k
    return kern

def procedural_metal(h, w, angle_deg=0.0, ksize=31):
    # base noise
    base = (np.random.rand(h, w) * 255).astype(np.uint8)
    kernel = _dir_kernel(angle_deg, ksize)
    brushed = cv2.filter2D(base, -1, kernel)

    # mild vignetting
    y, x = np.indices((h, w))
    cy, cx = h/2.0, w/2.0
    r = np.sqrt(((y - cy)/cy)**2 + ((x - cx)/cx)**2)
    vignette = np.clip(1.0 - 0.6 * r, 0, 1)
    out = (brushed.astype(np.float32) * (0.7 + 0.3*vignette)).astype(np.uint8)
    return out

def compose_background(H, W, cfg, tex_cache):
    # Decide orientation
    angle = np.random.uniform(cfg.orient_deg_range[0], cfg.orient_deg_range[1]) if np.random.rand() < cfg.orient_blur_prob else 0.0
    proc = procedural_metal(H, W, angle_deg=angle, ksize=cfg.orient_blur_kernel)
    if tex_cache:
        if np.random.rand() < cfg.use_real_textures_prob:
            tile = sample_texture(tex_cache, H, W)
        else:
            tile = None
    else:
        tile = None

    if tile is not None:
        # Normalize and blend real texture with procedural
        tile_f = tile.astype(np.float32) / 255.0
        proc_f = proc.astype(np.float32) / 255.0
        bg = np.clip(cfg.texture_blend * tile_f + (1 - cfg.texture_blend) * proc_f, 0, 1)
        bg = (bg * 255).astype(np.uint8)
    else:
        bg = proc

    return bg

def add_glare_multi(img, layers=2, strength_range=(0.3, 0.8), width_range=(8.0, 25.0), angle_jitter=30.0):
    h, w = img.shape
    out = img.astype(np.float32) / 255.0
    base_angle = np.random.uniform(-30, 30)
    for _ in range(layers):
        ang = base_angle + np.random.uniform(-angle_jitter, angle_jitter)
        y, x = np.indices((h, w))
        angle = np.deg2rad(ang)
        xr = (x - w/2) * np.cos(angle) + (y - h/2) * np.sin(angle)
        width = np.random.uniform(width_range[0], width_range[1])
        stripe = np.exp(-(xr**2) / (2*(width**2)))
        stripe /= stripe.max() + 1e-6
        s = np.random.uniform(strength_range[0], strength_range[1])
        out = np.clip(out + s * stripe, 0, 1)
    return (out * 255).astype(np.uint8)

def add_scratches(img, count=50, length=40, thickness=1):
    out = img.copy()
    h, w = img.shape
    for _ in range(count):
        x0 = np.random.randint(0, w)
        y0 = np.random.randint(0, h)
        angle = np.random.rand() * np.pi
        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))
        cv2.line(out, (x0, y0), (x1, y1), int(np.random.randint(180, 255)), thickness)
    return out
