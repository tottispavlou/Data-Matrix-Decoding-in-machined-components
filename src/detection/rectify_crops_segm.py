import argparse
from pathlib import Path
import numpy as np
import cv2


# -------------------- Label utils --------------------

def parse_line(line: str):
    """
    YOLO segmentation line:
        class x1 y1 x2 y2 ... xn yn [conf]
    Returns (cls, pts[N,2], conf) or None
    """
    parts = line.strip().split()
    if len(parts) < 9:  # need at least 4 points
        return None
    cls = int(float(parts[0]))
    nums = list(map(float, parts[1:]))

    # Optional trailing confidence
    if len(nums) % 2 == 1:
        conf = float(nums[-1])
        nums = nums[:-1]
    else:
        conf = 1.0

    if len(nums) < 8:
        return None

    pts = np.array(nums, dtype=np.float32).reshape(-1, 2)
    return cls, pts, conf


def load_labels(label_path: Path):
    items = []
    with open(label_path, "r") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                items.append(parsed)
    return items


def to_pixels(pts: np.ndarray, W: int, H: int) -> np.ndarray:
    """Convert normalized segmentation coords (0–1) to pixels if needed."""
    pts = pts.copy()
    if np.max(pts) <= 1.5:  # treat as normalized
        pts[:, 0] *= W
        pts[:, 1] *= H
    return pts


# -------------------- Geometry helpers --------------------

def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise: TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    rect[1] = pts[np.argmin(d)]      # TR
    rect[3] = pts[np.argmax(d)]      # BL
    return rect

def find_mask_corners(poly_px: np.ndarray, H: int, W: int):
    """
    Find 4 extreme corner points (TL, TR, BR, BL) from the mask polygon.
    Simple manual method using min/max coordinate combinations.
    """
    # ensure inside image
    poly_px = np.clip(poly_px, [0, 0], [W - 1, H - 1])

    # compute helper metrics
    s = poly_px.sum(axis=1)       # x + y
    diff = poly_px[:, 0] - poly_px[:, 1]  # x - y

    tl = poly_px[np.argmin(s)]    # smallest x+y
    br = poly_px[np.argmax(s)]    # largest x+y
    tr = poly_px[np.argmax(diff)] # largest x−y
    bl = poly_px[np.argmin(diff)] # smallest x−y

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    return corners, None

def perspective_warp(image: np.ndarray, src_pts: np.ndarray, force_square: bool = False):
    """
    Perspective-correct crop given 4 pixel points (TL,TR,BR,BL).
    If force_square=True, make output a square using the longer side.
    """
    (tl, tr, br, bl) = src_pts.astype(np.float32)

    # Compute target size
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    Wd = float(max(widthA, widthB))
    Hd = float(max(heightA, heightB))

    if force_square:
        L = int(round(max(Wd, Hd)))
        Wout, Hout = L, L
        dst = np.array([[0, 0],
                        [L - 1, 0],
                        [L - 1, L - 1],
                        [0, L - 1]], dtype=np.float32)
    else:
        Wout = max(1, int(round(Wd)))
        Hout = max(1, int(round(Hd)))
        dst = np.array([[0, 0],
                        [Wout - 1, 0],
                        [Wout - 1, Hout - 1],
                        [0, Hout - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, M, (Wout, Hout))
    return warped


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Rectify and crop from YOLO segmentation masks.")
    ap.add_argument("--img_dir", type=str, required=True, help="Directory of input images")
    ap.add_argument("--label_dir", type=str, required=True, help="Directory with YOLO segmentation labels")
    ap.add_argument("--out_dir", type=str, default="rectified_crops", help="Output directory")
    ap.add_argument("--min_conf", type=float, default=0.0, help="Minimum confidence to accept (if present)")
    ap.add_argument("--square", action="store_true", help="Force rectified crop to be square (uses longer side)")
    ap.add_argument("--debug", action="store_true", help="Save debug overlays")
    args = ap.parse_args()

    img_dir, label_dir, out_dir = Path(args.img_dir), Path(args.label_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (out_dir / "_debug").mkdir(exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    img_paths = sorted([p for e in exts for p in img_dir.glob(e)])

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

        # Choose most confident polygon (or first if all conf==1)
        cls, pts_norm, conf = max(labels, key=lambda x: x[2])
        if conf < args.min_conf:
            continue

        # 1) normalize -> PIXELS
        poly_px = to_pixels(pts_norm, W, H)

        # 2) find corners from the MASK (TL,TR,BR,BL) in PIXELS
        corners, mask = find_mask_corners(poly_px, H, W)
        if corners is None:
            continue

        # 3) perspective warp with those pixel corners
        crop = perspective_warp(img, corners, force_square=args.square)

        # Save crop
        out_path = out_dir / f"{img_path.stem}_rectified.png"
        cv2.imwrite(str(out_path), crop)
        saved += 1

        # Debug overlay: cyan = mask outline, red = used crop rectangle
        if args.debug:
            dbg = img.copy()
            # draw mask outline
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(dbg, contours, -1, (255, 255, 0), 2)
            # draw rectangle used for warp
            cv2.polylines(dbg, [corners.astype(np.int32)], True, (0, 0, 255), 2)
            cv2.putText(dbg, f"conf={conf:.2f}", tuple(np.int32(corners.mean(axis=0))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(out_dir / "_debug" / f"{img_path.stem}_dbg.jpg"), dbg)

    print(f"[OK] Saved {saved} rectified crops to {out_dir}")


if __name__ == "__main__":
    main()
