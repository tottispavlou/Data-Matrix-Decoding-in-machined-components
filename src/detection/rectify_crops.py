import argparse
from pathlib import Path
import numpy as np
import cv2

def parse_line(line):
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    cls = int(float(parts[0]))
    coords = list(map(float, parts[1:9]))  # 8 coords
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    return cls, pts

def load_labels(label_path):
    quads = []
    with open(label_path, "r") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                quads.append(parsed)
    return quads

def to_pixels(pts, W, H):
    if np.max(pts) <= 1.5:  # normalized
        pts[:, 0] *= W
        pts[:, 1] *= H
    return pts

def pad_quad(quad, pad_ratio=0.1):
    """
    Expand quad by moving each point away from its centroid.
    """
    cx, cy = quad.mean(axis=0)
    padded = []
    for (x, y) in quad:
        dx, dy = x - cx, y - cy
        x_new = cx + dx * (1 + pad_ratio)
        y_new = cy + dy * (1 + pad_ratio)
        padded.append([x_new, y_new])
    return np.array(padded, dtype=np.float32)

def warp_quad(image, quad, out_size=256):
    dst = np.array([
        [0, 0],
        [out_size-1, 0],
        [out_size-1, out_size-1],
        [0, out_size-1]
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, H, (out_size, out_size))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--label_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="rectified_crops")
    ap.add_argument("--pad", type=float, default=0.1, help="Padding ratio")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img_dir, label_dir, out_dir = Path(args.img_dir), Path(args.label_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (out_dir / "_debug").mkdir(exist_ok=True)

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png")))
    saved = 0
    for img_path in img_paths:
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        for k, (cls, pts) in enumerate(load_labels(label_path)):
            quad_px = to_pixels(pts.copy(), W, H)
            quad_padded = pad_quad(quad_px, pad_ratio=args.pad)
            crop = warp_quad(img, quad_padded, out_size=args.size)

            out_path = out_dir / f"{img_path.stem}_det{k}.png"
            cv2.imwrite(str(out_path), crop)
            saved += 1

            if args.debug:
                dbg = img.copy()
                cv2.polylines(dbg, [quad_px.astype(int)], True, (0, 255, 0), 2)
                cv2.polylines(dbg, [quad_padded.astype(int)], True, (0, 0, 255), 2)
                dbg_out = out_dir / "_debug" / f"{img_path.stem}_det{k}.jpg"
                cv2.imwrite(str(dbg_out), dbg)

    print(f"[OK] Saved {saved} crops to {out_dir}")

if __name__ == "__main__":
    main()
