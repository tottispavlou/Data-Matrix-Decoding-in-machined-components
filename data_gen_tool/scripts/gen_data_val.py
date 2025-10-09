#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import argparse

def draw_yolo_obb_overlay(img_path: Path, label_path: Path, out_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        return
    H, W = img.shape[:2]

    # Load YOLO label
    try:
        with open(label_path, "r") as f:
            parts = f.readline().strip().split()
        if len(parts) != 9:
            print(f"[WARN] Bad label format: {label_path}")
            return
        cls_id = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
    except Exception as e:
        print(f"[ERR] Reading {label_path}: {e}")
        return

    # Denormalize coords to pixels
    coords[:, 0] *= W
    coords[:, 1] *= H
    coords = coords.astype(int)

    # Draw polygon overlay
    overlay = img.copy()
    cv2.polylines(overlay, [coords], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, (x, y) in enumerate(coords):
        cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(overlay, str(i+1), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # Save overlay
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), overlay)
    print(f"[OK] Saved overlay: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Visualize YOLO OBB labels on images.")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--out_dir", default="overlay_check")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]

    print(f"[INFO] Found {len(imgs)} images in {img_dir}")
    for img_path in imgs:
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"[WARN] Missing label for {img_path.name}")
            continue
        draw_yolo_obb_overlay(img_path, label_path, out_dir)

    print(f"[DONE] Overlays saved in {out_dir}")

if __name__ == "__main__":
    main()
