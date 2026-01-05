import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.feature import blob_log


# ============================================================
# 1. LABELS
# ============================================================

def find_label_for_image(img_path: Path, label_dir: Path) -> Path:
    """IMG_rectified.png -> IMG.txt"""
    stem = img_path.stem.replace("_rectified", "")
    return label_dir / f"{stem}.txt"


def read_top_label(label_path: Path):
    """Return (class_id, conf) of highest-confidence line."""
    if not label_path.exists():
        return None
    with open(label_path, "r") as f:
        lines = [l.strip().split() for l in f if l.strip()]
    if not lines:
        return None
    # assume: cls cx cy w h conf OR cls ... conf
    pairs = [(int(float(l[0])), float(l[-1])) for l in lines]
    return max(pairs, key=lambda x: x[1])


# ============================================================
# 2. VALLEY-BASED DOTMAP (PER QUADRANT)
# ============================================================

def _choose_dot_mask(bg_mask: np.ndarray, dot_mask: np.ndarray) -> np.ndarray:
    """
    Decide which mask corresponds to dots by CC area.
    Dots → many small blobs; background → large regions.
    """
    def median_area(mask):
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num <= 1:
            return 1e9
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return 1e9
        return float(np.median(areas))

    A = median_area(bg_mask)
    B = median_area(dot_mask)
    # dots = mask with smaller typical component area
    return bg_mask if A < B else dot_mask


def valley_dot_mask(gray_quad: np.ndarray) -> np.ndarray:
    """
    Histogram valley between two tallest peaks.
    Returns binary mask with dots=255, bg=0.
    """
    q = gray_quad.astype(np.uint8)
    hist, _ = np.histogram(q.ravel(), bins=256, range=(0, 256))
    peaks, _ = find_peaks(hist, distance=10)

    if len(peaks) < 2:
        _, bw = cv2.threshold(q, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw

    top2 = peaks[np.argsort(hist[peaks])[-2:]]
    p1, p2 = sorted(top2)
    valley_rel = np.argmin(hist[p1:p2])
    valley = p1 + valley_rel

    bg_mask = (q <= valley).astype(np.uint8) * 255
    dot_mask = (q >= valley).astype(np.uint8) * 255

    dots = _choose_dot_mask(bg_mask, dot_mask)

    return dots


def make_dotmap_valley(gray: np.ndarray) -> np.ndarray:
    """
    Split image into 4 quadrants, valley mapping per quad,
    then stitch back and clean.
    """
    h, w = gray.shape
    hh, ww = h // 2, w // 2

    q1 = gray[0:hh, 0:ww]
    q2 = gray[0:hh, ww:w]
    q3 = gray[hh:h, 0:ww]
    q4 = gray[hh:h, ww:w]

    m1 = valley_dot_mask(q1)
    m2 = valley_dot_mask(q2)
    m3 = valley_dot_mask(q3)
    m4 = valley_dot_mask(q4)

    top = np.hstack([m1, m2])
    bot = np.hstack([m3, m4])
    dotmap = np.vstack([top, bot])

    # global clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dotmap = cv2.morphologyEx(dotmap, cv2.MORPH_OPEN, kernel)
    dotmap = cv2.morphologyEx(dotmap, cv2.MORPH_CLOSE, kernel)
    return dotmap


# ============================================================
# 3. LoG BLOB DETECTION ON DOTMAP
# ============================================================

def detect_dots_log(dotmap: np.ndarray) -> np.ndarray:
    """
    Run LoG on the binary dotmap to get dot centers.
    Returns array of (x, y) in image coordinates.
    """
    h, w = dotmap.shape
    img = dotmap.astype(np.float32) / 255.0

    # rough cell size (between 14 and 16 modules)
    cell = min(h, w) / 15.0
    dot_radius = 0.35 * cell
    sigma = dot_radius / np.sqrt(2)

    min_sigma = max(0.5, sigma * 0.8)
    max_sigma = sigma * 1.2

    blobs = blob_log(
        img,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=5,
        threshold=0.02,
    )

    if blobs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # blob_log returns (y, x, sigma)
    centers = []
    for (y, x, s) in blobs:
        r = s * np.sqrt(2.0)   # approximate radius
        if 0.2 * cell < r < 0.8 * cell:
            centers.append((float(x), float(y)))

    return np.array(centers, dtype=np.float32)


def draw_dots_on_map(dotmap: np.ndarray, dots: np.ndarray) -> np.ndarray:
    """
    Debug: draw red circles on the dotmap (not on original).
    """
    vis = cv2.cvtColor(dotmap, cv2.COLOR_GRAY2BGR)
    for (x, y) in dots:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    return vis


# ============================================================
# 4. GRID FITTING (ROW/COLUMN CLUSTERING)
# ============================================================

def cluster_1d(values: np.ndarray, est_step: float, ratio: float = 0.5):
    """
    Simple gap-based 1D clustering; returns cluster centers.
    """
    if len(values) == 0:
        return np.array([])

    vals = np.sort(values)
    clusters = [[vals[0]]]
    thresh = est_step * ratio

    for v in vals[1:]:
        if abs(v - np.mean(clusters[-1])) <= thresh:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    centers = np.array([np.mean(c) for c in clusters], dtype=np.float32)
    return centers


def fit_grid_from_dots(dots: np.ndarray, img_shape) -> tuple | None:
    """
    Cluster dots into horizontal rows and vertical columns.
    Infer N ~ 14 or 16.
    Returns (row_centers, col_centers, N) or None.
    """
    if dots.shape[0] < 10:
        return None

    h, w = img_shape
    xs = dots[:, 0]
    ys = dots[:, 1]

    est_step_y = (ys.max() - ys.min()) / 15.0 if ys.max() > ys.min() else h / 15.0
    est_step_x = (xs.max() - xs.min()) / 15.0 if xs.max() > xs.min() else w / 15.0

    row_c = cluster_1d(ys, est_step_y)
    col_c = cluster_1d(xs, est_step_x)

    if len(row_c) < 5 or len(col_c) < 5:
        return None

    approx_N = int(round((len(row_c) + len(col_c)) / 2))
    N = 14 if abs(approx_N - 14) <= abs(approx_N - 16) else 16

    row_c = np.sort(row_c)
    col_c = np.sort(col_c)

    def trim_to_N(arr, N):
        if len(arr) <= N:
            return arr
        mid = len(arr) // 2
        halfN = N // 2
        start = max(0, mid - halfN)
        end = start + N
        return arr[start:end]

    row_c = trim_to_N(row_c, N)
    col_c = trim_to_N(col_c, N)

    if len(row_c) != N or len(col_c) != N:
        return None

    return row_c, col_c, N


def draw_grid_on_map(dotmap: np.ndarray, row_c: np.ndarray, col_c: np.ndarray) -> np.ndarray:
    """
    Debug: draw row/column lines on the dotmap.
    """
    vis = cv2.cvtColor(dotmap, cv2.COLOR_GRAY2BGR)
    h, w = dotmap.shape
    for y in row_c:
        cv2.line(vis, (0, int(y)), (w - 1, int(y)), (0, 0, 255), 1, lineType=cv2.LINE_AA)
    for x in col_c:
        cv2.line(vis, (int(x), 0), (int(x), h - 1), (0, 0, 255), 1, lineType=cv2.LINE_AA)
    return vis


# ============================================================
# 5. BUILD GRID + ORIENT USING L-RULES
# ============================================================

def build_raw_grid(dots: np.ndarray, row_c: np.ndarray, col_c: np.ndarray, N: int) -> np.ndarray:
    """
    N x N grid; 1 if a dot is near intersection, else 0.
    """
    grid = np.zeros((N, N), dtype=np.uint8)
    if dots.shape[0] == 0:
        return grid

    row_c_sorted = np.sort(row_c)
    col_c_sorted = np.sort(col_c)

    step_y = np.median(np.diff(row_c_sorted)) if len(row_c_sorted) > 1 else 1.0
    step_x = np.median(np.diff(col_c_sorted)) if len(col_c_sorted) > 1 else 1.0
    thresh = 0.5 * min(step_x, step_y)

    for r, y in enumerate(row_c_sorted):
        for c, x in enumerate(col_c_sorted):
            d = np.sqrt((dots[:, 0] - x) ** 2 + (dots[:, 1] - y) ** 2)
            if np.min(d) < thresh:
                grid[r, c] = 1
    return grid


def _side_stats(arr: np.ndarray):
    """Return (count, longest_run, alternation_ratio)."""
    N = len(arr)
    count = int(arr.sum())
    # longest run of ones
    best = cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    # alternation ratio (for timing pattern if needed)
    if N <= 1:
        alt = 0.0
    else:
        alt = sum(arr[i] != arr[i+1] for i in range(N-1)) / (N-1)
    return count, best, alt


def orient_grid_with_L(grid: np.ndarray) -> np.ndarray:
    """
    Rotate grid so that the solid L is on left+bottom sides.
    Uses:
      - full-length dots,
      - >= N/2 dots,
      - dot-run consistency.
    """
    N = grid.shape[0]

    def score_rotation(g):
        top = g[0, :]
        bottom = g[-1, :]
        left = g[:, 0]
        right = g[:, -1]

        def is_L_side(arr):
            count, run, _ = _side_stats(arr)
            run_ratio = run / N
            mean_occ = count / N
            full = (count >= N - 1)
            half = (count >= N // 2)
            dot_run = (run_ratio > 0.7) or (mean_occ > 0.6)
            return full or half or dot_run

        L_top = is_L_side(top)
        L_bottom = is_L_side(bottom)
        L_left = is_L_side(left)
        L_right = is_L_side(right)

        # want left+bottom to be L, top+right not
        score = 0
        if L_left:
            score += 2
        if L_bottom:
            score += 2
        if L_top:
            score -= 1
        if L_right:
            score -= 1

        # bonus if we have exactly 2 L sides and they meet at corner
        L_sides = [L_top, L_right, L_bottom, L_left]  # clockwise
        if sum(L_sides) == 2:
            # check for consecutive pair
            for i in range(4):
                if L_sides[i] and L_sides[(i+1) % 4]:
                    score += 2
                    break

        return score

    best_score = -1e9
    best = grid.copy()
    for k in range(4):
        g = np.rot90(grid, k)
        s = score_rotation(g)
        if s > best_score:
            best_score = s
            best = g
    return best


def grid_to_image(grid: np.ndarray, scale: int = 10) -> np.ndarray:
    """
    Convert binary grid (1=dot, 0=empty) to synthetic black/white DMC image.
    1 -> black, 0 -> white.
    """
    small = (1 - grid.astype(np.uint8)) * 255  # invert: dot=black
    N = small.shape[0]
    img = cv2.resize(
        small,
        (N * scale, N * scale),
        interpolation=cv2.INTER_NEAREST
    )
    return img


# ============================================================
# 6. LASER PIPELINE
# ============================================================

def process_laser(gray: np.ndarray) -> np.ndarray:
    """Simple local thresholding for laser-etched codes."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    thr = cv2.adaptiveThreshold(
        cl, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    thr = cv2.morphologyEx(
        thr,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return thr


# ============================================================
# 7. MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True, help="Folder with rectified crops")
    ap.add_argument("--labels", required=True, help="YOLO label folder")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--debug", action="store_true", help="Save debug images")
    args = ap.parse_args()

    img_dir = Path(args.imgs)
    label_dir = Path(args.labels)
    out_dir = Path(args.out)
    dbg_dir = out_dir / "_debug"

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.png"))

    for img_path in img_paths:
        # load
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        h, w = gray.shape

        # label
        label_path = find_label_for_image(img_path, label_dir)
        label = read_top_label(label_path) if label_path else None
        if label is None:
            continue
        cls_id, conf = label
        mode = "dotpeen" if cls_id == 0 else "laser" if cls_id == 1 else "unknown"

        print(f"[INFO] {img_path.name}: {mode} (conf={conf:.3f})")

        # LASER
        if mode == "laser":
            thr = process_laser(gray)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_laser.png"), thr)
            if args.debug:
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_laser_debug.png"), thr)
            continue

        # DOT-PEENED
        if mode != "dotpeen":
            continue

        # Step 1: dotmap
        dotmap = make_dotmap_valley(gray)
        if args.debug:
            cv2.imwrite(str(dbg_dir / f"{img_path.stem}_step1_dotmap.png"), dotmap)

        # Step 2: LoG dots
        dots = detect_dots_log(dotmap)
        if args.debug:
            vis_dots = draw_dots_on_map(dotmap, dots)
            cv2.imwrite(str(dbg_dir / f"{img_path.stem}_step2_dots.png"), vis_dots)

        if dots.shape[0] < 10:
            print(f"[WARN] Too few dots in {img_path.name}")
            continue

        # Step 3: grid fit
        grid_info = fit_grid_from_dots(dots, (h, w))
        if grid_info is None:
            print(f"[WARN] Grid fitting failed for {img_path.name}")
            continue
        row_c, col_c, N = grid_info
        if args.debug:
            vis_grid = draw_grid_on_map(dotmap, row_c, col_c)
            cv2.imwrite(str(dbg_dir / f"{img_path.stem}_step3_grid.png"), vis_grid)

        # Step 4: raw grid
        raw_grid = build_raw_grid(dots, row_c, col_c, N)

        # Step 5: orient with L on left+bottom
        oriented = orient_grid_with_L(raw_grid)

        # Step 6: synthetic DMC image
        syn_img = grid_to_image(oriented, scale=12)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_synthetic.png"), syn_img)

    print("[OK] Done.")


if __name__ == "__main__":
    main()
