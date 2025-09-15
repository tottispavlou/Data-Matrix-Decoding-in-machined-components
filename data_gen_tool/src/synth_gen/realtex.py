
import os
from pathlib import Path
import numpy as np
import cv2
import random
from typing import List, Optional

def load_texture_paths(folder: Optional[str]) -> List[Path]:
    if not folder:
        return []
    p = Path(folder)
    if not p.exists():
        return []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    return [q for q in p.rglob('*') if q.suffix.lower() in exts]

def sample_texture(text_paths: List[Path], H: int, W: int) -> Optional[np.ndarray]:
    if not text_paths:
        return None
    path = random.choice(text_paths)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    if h < H or w < W:
        # resize up minimally
        scale = max(H/h, W/w)
        img = cv2.resize(img, (int(w*scale)+1, int(h*scale)+1), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape
    # random crop
    y0 = random.randint(0, h - H)
    x0 = random.randint(0, w - W)
    tile = img[y0:y0+H, x0:x0+W]
    return tile
