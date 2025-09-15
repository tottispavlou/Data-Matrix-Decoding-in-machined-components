
import numpy as np
import cv2

def warp_perspective(img, strength=0.12):
    """Apply a mild random perspective warp to simulate unrectified scenes."""
    h, w = img.shape[:2]
    # source quad
    src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    # destination quad with random perturbation
    jitter = strength * min(h,w)
    dst = src + np.random.uniform(-jitter, jitter, size=src.shape).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped, H

def yolo_box_for_centered(img, pad=0):
    h, w = img.shape[:2]
    x = pad
    y = pad
    bw = w - 2*pad
    bh = h - 2*pad
    # YOLO txt format relative (class cx cy w h)
    cx = (x + bw/2) / w
    cy = (y + bh/2) / h
    ww = bw / w
    hh = bh / h
    return (cx, cy, ww, hh)
