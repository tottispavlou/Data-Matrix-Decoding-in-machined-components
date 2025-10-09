import argparse
from pathlib import Path
import numpy as np
import cv2

def parse_line(line):
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    cls = int(float(parts[0]))
    coords = list(map(float, parts[1:9])) # last is confidence
    conf = float(parts[9]) if len(parts) >= 10 else 1.0
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    return cls, pts, conf

def load_labels(label_path):
    quads = []
    with open(label_path, "r") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                quads.append(parsed)  # (cls, pts, conf)
    return quads

def to_pixels(pts, W, H):
    # if normalized, convert to pixel coords
    if np.max(pts) <= 1.5:
        pts[:, 0] *= W
        pts[:, 1] *= H
    return pts

def pad_quad(quad, pad_ratio=0.1):
    """Expand quad by moving each point away from its centroid."""
    cx, cy = quad.mean(axis=0)
    d = quad - np.array([cx, cy], dtype=np.float32)
    return quad + d * pad_ratio

def warp_quad(image, quad, out_size=256):
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
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
    ap.add_argument("--min_conf", type=float, default=0.0, help="Minimum confidence to accept")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img_dir, label_dir, out_dir = Path(args.img_dir), Path(args.label_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (out_dir / "_debug").mkdir(exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    img_paths = []
    for e in exts:
        img_paths.extend(img_dir.glob(e))
    img_paths = sorted(img_paths)

    saved = 0
    for img_path in img_paths:
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        labels = load_labels(label_path)
        if not labels:
            continue

        # pick most confident detection
        cls, pts, conf = max(labels, key=lambda x: x[2])
        if conf < args.min_conf:
            continue

        quad_px = to_pixels(pts.copy(), W, H)
        quad_padded = pad_quad(quad_px, pad_ratio=args.pad)
        crop = warp_quad(img, quad_padded, out_size=args.size)

        out_path = out_dir / f"{img_path.stem}_best.png"
        cv2.imwrite(str(out_path), crop)
        saved += 1

        if args.debug:
            dbg = img.copy()
            cv2.polylines(dbg, [quad_px.astype(int)], True, (0, 255, 0), 2)
            cv2.polylines(dbg, [quad_padded.astype(int)], True, (0, 0, 255), 2)
            cv2.putText(dbg, f"conf={conf:.2f}", tuple(quad_px.mean(axis=0).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            dbg_out = out_dir / "_debug" / f"{img_path.stem}_best.jpg"
            cv2.imwrite(str(dbg_out), dbg)

    print(f"[OK] Saved {saved} crops to {out_dir}")

if __name__ == "__main__":
    main()
