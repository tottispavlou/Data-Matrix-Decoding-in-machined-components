import argparse
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


# -----------------------------------------------------------
#  YOLO LABEL UTILITIES
# -----------------------------------------------------------

def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) < 9:
        return None

    cls = int(float(parts[0]))
    nums = list(map(float, parts[1:]))

    # optional trailing conf
    if len(nums) % 2 == 1:
        conf = nums[-1]
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
            p = parse_line(line)
            if p:
                items.append(p)
    return items


def to_pixels(pts, W, H):
    pts = pts.copy()
    if np.max(pts) <= 1.5:  # normalized YOLO coords
        pts[:, 0] *= W
        pts[:, 1] *= H
    return pts


# -----------------------------------------------------------
#  GEOMETRY HELPERS
# -----------------------------------------------------------

def simplify_contour(contour, eps_ratio=0.01):
    peri = cv2.arcLength(contour, True)
    eps = eps_ratio * peri
    return cv2.approxPolyDP(contour, eps, True)


def corner_angle(a, b, c):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c = c.astype(np.float32)
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    cosang = float(np.dot(ab, cb)) / denom
    cosang = max(-1.0, min(1.0, cosang))
    return np.arccos(cosang)


def pick_best_four_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    N = len(pts)
    if N < 4:
        return None
    if N == 4:
        return pts

    angles = []
    for i in range(N):
        a = pts[(i - 1) % N]
        b = pts[i]
        c = pts[(i + 1) % N]
        ang = corner_angle(a, b, c)
        angles.append((ang, i))

    angles.sort(key=lambda x: x[0])
    best_idxs = sorted([idx for _, idx in angles[:4]])
    return pts[best_idxs]


def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)

    # centroid
    c = pts.mean(axis=0)

    # angle of each point around centroid
    angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])

    # sort counter-clockwise
    order = np.argsort(angles)
    pts = pts[order]

    # rotate so top-left is first
    idx = np.argmin(pts[:,0] + pts[:,1])
    pts = np.roll(pts, -idx, axis=0)

    return pts



def warp_to_square(img, corners, out_size=400):
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (out_size, out_size), flags=cv2.INTER_NEAREST)

def expand_quad(quad, img_shape, pad_frac):
    """
    Expand a quad outward from its center by pad_frac.
    pad_frac is relative to quad size.
    """
    quad = quad.astype(np.float32)

    center = quad.mean(axis=0)
    vecs = quad - center

    quad_exp = center + (1.0 + pad_frac) * vecs

    h, w = img_shape[:2]
    quad_exp[:, 0] = np.clip(quad_exp[:, 0], 0, w - 1)
    quad_exp[:, 1] = np.clip(quad_exp[:, 1], 0, h - 1)

    return quad_exp

# -----------------------------------------------------------
#  MAIN
# -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Full DMC pipeline with ALL debug steps")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--out_dir", default="rectified_crops")
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--reso", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    lbl_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        (out_dir / "_debug").mkdir(exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    imgs = sorted([p for e in exts for p in img_dir.glob(e)])

    saved = 0
    reso_rows = []

    for img_path in imgs:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        H, W = img.shape[:2]
        labels = load_labels(label_path)
        if not labels:
            continue

        cls, pts_norm, conf = max(labels, key=lambda x: x[2])
        if conf < args.min_conf:
            continue

        # ----------------------
        # STEP 1: RAW MASK
        # ----------------------
        pts_px = to_pixels(pts_norm, W, H)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_px.astype(np.int32)], 255)

        # ----------------------
        # STEP 2: MORPH CLEAN + BRIDGE BREAK
        # ----------------------

        # Break thin "tails"/bridges that connect junk to the main region
        k_open = np.ones((3, 3), np.uint8)
        mask_break = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1) # 1 teration because the bridge is 1 bit usually

        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_break, connectivity=8)

        if n <= 1:
            continue  # no foreground

        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.argmax(areas)

        mask_main = np.zeros_like(mask_break)
        mask_main[lab == keep] = 255

        # Restore any tiny gaps introduced by opening
        k_close = np.ones((3, 3), np.uint8)
        mask_main = cv2.morphologyEx(mask_main, cv2.MORPH_CLOSE, k_close, iterations=1)

        # ----------------------
        # STEP 3: CONTOUR 
        # ----------------------
        cnts, _ = cv2.findContours(mask_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        if args.reso:
            code_area = cv2.contourArea(cnt)
            image_area = float(H * W)
            reso_metric = code_area / image_area

            reso_rows.append({
                "image_name": img_path.name,
                "image_width_px": W,
                "image_height_px": H,
                "image_area_px2": image_area,
                "code_area_px2": code_area,
                "code_ratio": reso_metric,
                "class": cls,
                "num_contour_pts": len(cnt),
            })

        # ----------------------
        # STEP 4: CONVEX HULL
        # ----------------------
        hull = cv2.convexHull(cnt)

        # ----------------------
        # STEP 5: DP SIMPLIFICATION
        # ----------------------
        hull_simpl = simplify_contour(hull, eps_ratio=0.01)
        pts_simpl = hull_simpl.reshape(-1, 2)

        # ----------------------
        # STEP 6: CORNER SELECTION
        # ----------------------
        pts4 = pick_best_four_corners(pts_simpl)
        if pts4 is None or len(pts4) != 4:
            print(f"[WARN] Could not detect 4 corners for {img_path.name}")
            continue

        # ----------------------
        # STEP 7: ORDERING
        # ----------------------
        corners = order_corners(pts4)

        if cls == 0:
            corners = expand_quad(corners, img.shape, pad_frac=0.03)
        if cls == 1:
            corners = expand_quad(corners, img.shape, pad_frac=0.1)

        # ----------------------
        # STEP 8: WARP
        # ----------------------
        crop = warp_to_square(img, corners, out_size=240)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_rectified.png"), crop)
        saved += 1

        # ============================================================
        # DEBUG VISUALIZATION OF ALL STEPS
        # ============================================================
        if args.debug:
            # helper function to label images
            def label(im, text):
                im2 = im.copy()
                cv2.putText(im2, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,  
                            0.5, (0, 255, 0), 2)
                return im2

            # panel 1: raw mask on original
            dbg1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            dbg1 = label(dbg1, "1: Raw Mask")

            # panel 2: closed main mask
            dbg2 = cv2.cvtColor(mask_main, cv2.COLOR_GRAY2BGR)
            dbg2 = label(dbg2, "2: Morph Mask Cleaning")

            # panel 3: contour overlay
            dbg3 = img.copy()
            cv2.polylines(dbg3, [cnt.astype(np.int32)], True, (0, 200, 255), 2)
            dbg3 = label(dbg3, "3: Contour")

            # panel 4: hull
            dbg4 = img.copy()
            cv2.polylines(dbg4, [hull.astype(np.int32)], True, (0, 255, 255), 2)
            dbg4 = label(dbg4, "4: Convex Hull")

            # panel 5: DP simplified hull
            dbg5 = img.copy()
            cv2.polylines(dbg5, [pts_simpl.astype(np.int32)], True, (0, 255, 0), 2)
            dbg5 = label(dbg5, "5: DP Simplified")

            # panel 6: sharp corner selection
            dbg6 = img.copy()
            for p in pts4:
                cv2.circle(dbg6, tuple(p.astype(int)), 6, (255, 0, 0), -1)
            dbg6 = label(dbg6, "6: Sharp Corners")

            # panel 7: final quad
            dbg7 = img.copy()
            cv2.polylines(dbg7, [corners.astype(np.int32)], True, (0, 0, 255), 2)
            dbg7 = label(dbg7, "7: Ordered Quad")

            # panel 8: warp result
            dbg8 = crop.copy()
            dbg8 = label(dbg8, "8: Warp (240x240)")

            # resize each panel
            panels = [dbg1, dbg2, dbg3, dbg4, dbg5, dbg6, dbg7, dbg8]
            panels = [cv2.resize(p, (300, 300)) for p in panels]

            # stack into a 2-row grid
            row1 = cv2.hconcat(panels[:4])
            row2 = cv2.hconcat(panels[4:])
            full_debug = cv2.vconcat([row1, row2])

            cv2.imwrite(str(out_dir / "_debug" / f"{img_path.stem}_FULL_DEBUG.png"), full_debug)

    if args.reso and reso_rows:
        df = pd.DataFrame(reso_rows)
        excel_path = out_dir / "reso_metrics.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"[OK] Wrote resolution metrics to {excel_path}")

    print(f"[OK] Saved {saved} crops to {out_dir}")


if __name__ == "__main__":
    main()
