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
    conf = float(parts[9]) if len(parts) >= 10 else 1.0
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    return cls, pts, conf


def load_labels(label_path):
    quads = []
    with open(label_path, "r") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                quads.append(parsed)
    return quads


def to_pixels(pts, W, H):
    """Convert normalized OBB coords (0–1) to pixels."""
    if np.max(pts) <= 1.5:
        pts[:, 0] *= W
        pts[:, 1] *= H
    return pts


def order_quad_ccw_tl_first(pts: np.ndarray) -> np.ndarray:
    """Order quad points TL, TR, BR, BL (counterclockwise)."""
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    q = pts[np.argsort(ang)]
    tl = np.argmin(q[:, 1] + 0.01 * q[:, 0])
    q = np.roll(q, -tl, axis=0)
    area = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    if area < 0:
        q = q[[0, 3, 2, 1]]
    return q


def warp_from_obb_padded(
    image,
    quad,
    pad_ratio=0.1,
    force_long_side_horizontal=True,
    out_size=None,
    border=cv2.BORDER_REPLICATE
):
    """
    Rectify (de-warp) an oriented bounding box region.
    Produces an upright, square crop that is perpendicular and undistorted.
    """
    q = order_quad_ccw_tl_first(quad)
    w = np.linalg.norm(q[1] - q[0])
    h = np.linalg.norm(q[3] - q[0])
    if w < 1 or h < 1:
        return None

    # Rotate if vertical — ensures consistent orientation
    if force_long_side_horizontal and h > w:
        q = q[[3, 0, 1, 2]]
        w, h = h, w

    # Unit edge directions
    u = (q[1] - q[0]) / w
    v = (q[3] - q[0]) / h

    # Apply padding in source coords
    pad_x = pad_ratio * w
    pad_y = pad_ratio * h

    qpad = np.zeros_like(q)
    qpad[0] = q[0] - u * pad_x - v * pad_y  # TL
    qpad[1] = q[1] + u * pad_x - v * pad_y  # TR
    qpad[2] = q[2] + u * pad_x + v * pad_y  # BR
    qpad[3] = q[3] - u * pad_x + v * pad_y  # BL

    # Destination rectangle
    Wd = int(round(w + 2 * pad_x))
    Hd = int(round(h + 2 * pad_y))
    dst = np.array([[0, 0], [Wd - 1, 0], [Wd - 1, Hd - 1], [0, Hd - 1]], dtype=np.float32)

    # Perspective transform
    M = cv2.getPerspectiveTransform(qpad, dst)

    # Warp to rectified view
    warped = cv2.warpPerspective(image, M, (Wd, Hd), flags=cv2.INTER_LINEAR, borderMode=border)

    # Optional resizing
    if out_size is not None:
        warped = cv2.resize(warped, out_size, interpolation=cv2.INTER_AREA)

    return warped, qpad


# -------------------- Main Script --------------------

def main():
    ap = argparse.ArgumentParser(description="Rectify and crop OBB detections from YOLO outputs.")
    ap.add_argument("--img_dir", type=str, required=True, help="Directory of input images")
    ap.add_argument("--label_dir", type=str, required=True, help="Directory with YOLO OBB labels")
    ap.add_argument("--out_dir", type=str, default="rectified_crops", help="Output directory")
    ap.add_argument("--pad", type=float, default=0.1, help="Padding ratio around box")
    ap.add_argument("--size", type=int, default=256, help="Output crop size")
    ap.add_argument("--min_conf", type=float, default=0.0, help="Minimum confidence to accept")
    ap.add_argument("--debug", action="store_true", help="Save debug overlays")
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

        crop, qpad = warp_from_obb_padded(
            img,
            quad_px,
            pad_ratio=args.pad,
            force_long_side_horizontal=True,
            out_size=(args.size, args.size),
        )

        if crop is None:
            continue

        out_path = out_dir / f"{img_path.stem}_best.png"
        cv2.imwrite(str(out_path), crop)
        saved += 1

        # Debug visualization
        if args.debug:
            dbg = img.copy()
            cv2.polylines(dbg, [np.int32(qpad)], True, (0, 0, 255), 2)
            cv2.putText(dbg, f"conf={conf:.2f}", tuple(np.int32(qpad.mean(axis=0))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            dbg_out = out_dir / "_debug" / f"{img_path.stem}_dbg.jpg"
            cv2.imwrite(str(dbg_out), dbg)

    print(f"[OK] Saved {saved} rectified crops to {out_dir}")


if __name__ == "__main__":
    main()
