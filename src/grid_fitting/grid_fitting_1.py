# grid_fitting.py
# Dot-peened: quadrant polarity -> LoG blobs -> rectangular N-grid (14/16) -> per-cell filtering ->
#             choose L -> corner validation -> (optional) L-anchored regrid -> force borders -> synthetic
# Laser: simple adaptive threshold
#
# Debug outputs (if --debug):
#  step0:  *_step0_hists.png, *_step0_polarity.png
#  per N:  *_N{N}_step1_raw_grid_blobs.png (blue blobs + red cell boundaries)
#          *_N{N}_step2_filtered_grid_blobs.png (green blobs + red cell boundaries)
#          *_N{N}_step3_regrid.png (ONLY if recalibration triggered)
#
# Output:
#   *_N14_synthetic.png, *_N16_synthetic.png (dotpeen)
#   *_laser.png (laser)

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log


# ============================================================
# Labels
# ============================================================

def find_label_for_image(img_path: Path, label_dir: Path) -> Path:
    stem = img_path.stem.replace("_rectified", "")
    return label_dir / f"{stem}.txt"


def read_top_label(label_path: Path):
    if not label_path.exists():
        return None
    lines = []
    with open(label_path, "r") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            parts = l.split()
            lines.append(parts)
    if not lines:
        return None
    # assume last token is confidence, first is class
    pairs = [(int(float(p[0])), float(p[-1])) for p in lines]
    return max(pairs, key=lambda x: x[1])


# ============================================================
# Step 0: Quadrant polarity (peak + adjacency rule) + hist debug
# ============================================================

def quadrant_polarity_fix(gray: np.ndarray):
    """
    Candidate = quadrant histogram max peak is bright (>128).
    Invert ONLY if candidate has an adjacent candidate neighbor:
        TL–TR, TL–BL, TR–BR, BL–BR
    """
    h, w = gray.shape
    hh, ww = h // 2, w // 2
    Q = [
        gray[0:hh, 0:ww],   # 0 TL
        gray[0:hh, ww:w],   # 1 TR
        gray[hh:h, 0:ww],   # 2 BL
        gray[hh:h, ww:w],   # 3 BR
    ]

    cand = [False] * 4
    hists = []
    for i, q in enumerate(Q):
        hist = cv2.calcHist([q], [0], None, [256], [0, 256]).flatten()
        hists.append(hist)
        peak = int(np.argmax(hist))
        cand[i] = (peak > 128)

    adj_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
    invert_mask = [False] * 4
    for a, b in adj_pairs:
        if cand[a] and cand[b]:
            invert_mask[a] = True
            invert_mask[b] = True

    # isolated / diagonal -> no inversion
    if not any(invert_mask):
        invert_mask = [False] * 4

    fixed = []
    for i, q in enumerate(Q):
        fixed.append(255 - q if invert_mask[i] else q.copy())

    top = np.hstack([fixed[0], fixed[1]])
    bot = np.hstack([fixed[2], fixed[3]])
    out = np.vstack([top, bot])
    return out, hists


def save_hist_panel(hists, out_path: Path):
    plt.figure(figsize=(6, 6))
    titles = ["Q1", "Q2", "Q3", "Q4"]
    for i, hist in enumerate(hists):
        plt.subplot(2, 2, i + 1)
        plt.plot(hist)
        plt.title(titles[i])
        plt.xlim(0, 255)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


# ============================================================
# Step 1: LoG blobs
# ============================================================

def detect_dots_log(gray: np.ndarray,
                    grid_size_virtual: int = 15,
                    threshold: float = 0.049):
    """
    LoG tuned by a virtual ~15x15 pitch. (We later map to N=14 and N=16 grids.)
    Returns blobs (y,x,r), gray_used, cell_virtual
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    gray_used = gray.copy()
    h, w = gray_used.shape
    cell = min(h, w) / float(grid_size_virtual)

    dot_radius = 0.5 * cell
    sigma_est = dot_radius / np.sqrt(2.0)
    sigma_min = max(0.6, 0.65 * sigma_est)
    sigma_max = 1.15 * sigma_est

    img_norm = gray_used.astype(np.float32) / 255.0
    blobs = blob_log(
        img_norm,
        min_sigma=sigma_min,
        max_sigma=sigma_max,
        num_sigma=10,
        threshold=threshold,
        overlap=0.5
    )
    if blobs.size > 0:
        blobs[:, 2] *= np.sqrt(2.0)  # sigma -> radius

    # loose radius filter (relative to virtual cell)
    r_min = 0.25 * cell
    r_max = 1.10 * cell
    keep = [b for b in blobs if r_min <= b[2] <= r_max]
    if not keep:
        return np.zeros((0, 3), dtype=np.float32), gray_used, cell

    return np.array(keep, dtype=np.float32), gray_used, cell


# ============================================================
# Grid drawing (cell boundaries) + blob drawing
# ============================================================

def draw_cell_grid(vis: np.ndarray, N: int, color=(0, 0, 255), thickness=1):
    """Draw N×N cell boundaries (N+1 lines each direction)."""
    h, w = vis.shape[:2]
    step_x = w / float(N)
    step_y = h / float(N)

    # vertical boundaries
    for k in range(N + 1):
        x = int(round(k * step_x))
        cv2.line(vis, (x, 0), (x, h - 1), color, thickness)

    # horizontal boundaries
    for k in range(N + 1):
        y = int(round(k * step_y))
        cv2.line(vis, (0, y), (w - 1, y), color, thickness)


def draw_blobs(vis: np.ndarray, blobs: np.ndarray, color_bgr, thickness=2):
    for (y, x, r) in blobs:
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(r)), color_bgr, thickness, cv2.LINE_AA)


# ============================================================
# Step 2: per-cell filtering + spill check
# ============================================================

def mean_in_circle(gray: np.ndarray, cx: float, cy: float, r: float):
    """Fast-ish mean in disk via mask ROI."""
    h, w = gray.shape
    x0 = max(0, int(cx - r - 2))
    x1 = min(w, int(cx + r + 3))
    y0 = max(0, int(cy - r - 2))
    y1 = min(h, int(cy + r + 3))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[0:roi.shape[0], 0:roi.shape[1]]
    dx = (xx + x0) - cx
    dy = (yy + y0) - cy
    mask = (dx * dx + dy * dy) <= (r * r)
    vals = roi[mask]
    return float(np.mean(vals)) if vals.size else 0.0


def blob_contrast(gray: np.ndarray, cx: float, cy: float, r: float):
    """
    Contrast = |mean(disk r*0.6) - mean(ring [r*1.2..r*1.8])|
    Works for mixed bright/dark dots (just magnitude).
    """
    m_in = mean_in_circle(gray, cx, cy, max(1.0, 0.6 * r))
    m_r1 = mean_in_circle(gray, cx, cy, max(1.0, 1.8 * r))
    m_r0 = mean_in_circle(gray, cx, cy, max(1.0, 1.2 * r))
    # ring mean approx via difference (rough)
    m_ring = max(0.0, m_r1 - m_r0)
    return abs(m_in - m_ring)


def spill_fraction_sample(cx, cy, r, cell_bounds, samples=40):
    """
    Approx area spill using sampled points over disk.
    If >40% samples fall outside the cell rectangle -> reject.
    """
    x0, y0, x1, y1 = cell_bounds  # [x0,y0,x1,y1)
    outside = 0
    total = 0

    # sample rings: 0.5r, 0.8r, 1.0r
    radii = [0.5 * r, 0.8 * r, 1.0 * r]
    for rr in radii:
        for k in range(samples // len(radii)):
            ang = 2.0 * np.pi * (k / float(samples // len(radii)))
            px = cx + rr * np.cos(ang)
            py = cy + rr * np.sin(ang)
            total += 1
            if not (x0 <= px < x1 and y0 <= py < y1):
                outside += 1

    return outside / float(max(1, total))


def split_cells(N: int, h: int, w: int):
    """Return per-cell bounds: (i,j)->(x0,y0,x1,y1)."""
    step_x = w / float(N)
    step_y = h / float(N)
    bounds = {}
    for i in range(N):
        for j in range(N):
            x0 = j * step_x
            x1 = (j + 1) * step_x
            y0 = i * step_y
            y1 = (i + 1) * step_y
            bounds[(i, j)] = (x0, y0, x1, y1)
    return bounds, step_x, step_y


def assign_blobs_to_cells(blobs: np.ndarray, N: int, h: int, w: int):
    bounds, step_x, step_y = split_cells(N, h, w)
    per_cell = {(i, j): [] for i in range(N) for j in range(N)}
    for (y, x, r) in blobs:
        j = int(x / step_x)
        i = int(y / step_y)
        if 0 <= i < N and 0 <= j < N:
            per_cell[(i, j)].append((float(y), float(x), float(r)))
    return per_cell, bounds, step_x, step_y


def filter_blobs_in_cells(blobs: np.ndarray, gray_for_contrast: np.ndarray, N: int):
    """
    Rules:
      - center decides cell
      - reject if spill fraction > 0.40
      - if multiple blobs in cell: keep max(score = r * contrast)
      - also reject very small blobs (r < 0.30 * cellN)
    """
    h, w = gray_for_contrast.shape
    per_cell, bounds, step_x, step_y = assign_blobs_to_cells(blobs, N, h, w)
    cellN = min(h, w) / float(N)

    kept = []

    for (i, j), items in per_cell.items():
        if not items:
            continue
        x0, y0, x1, y1 = bounds[(i, j)]

        cand = []
        for (y, x, r) in items:
            if r < 0.30 * cellN:
                continue
            spill = spill_fraction_sample(x, y, r, (x0, y0, x1, y1), samples=48)
            if spill > 0.70:
                continue
            c = blob_contrast(gray_for_contrast, x, y, r)
            score = r * c
            cand.append((score, y, x, r))

        if not cand:
            continue

        best = max(cand, key=lambda t: t[0])
        kept.append((best[1], best[2], best[3]))

    if not kept:
        return np.zeros((0, 3), dtype=np.float32)

    return np.array(kept, dtype=np.float32)


def grid_from_blobs_rect(blobs: np.ndarray, N: int, h: int, w: int):
    """Occupancy grid by center-in-cell."""
    per_cell, _, _, _ = assign_blobs_to_cells(blobs, N, h, w)
    grid = np.zeros((N, N), dtype=np.uint8)
    for (i, j), items in per_cell.items():
        if items:
            grid[i, j] = 1
    return grid


# ============================================================
# L selection (strict) + border forcing (strict timing corner white)
# ============================================================

def choose_L_sides_strict(grid: np.ndarray):
    N = grid.shape[0]
    sides = {
        "top": grid[0, :],
        "right": grid[:, -1],
        "bottom": grid[-1, :],
        "left": grid[:, 0],
    }

    def stats(arr):
        count = int(arr.sum())
        best = cur = 0
        for v in arr:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        run_ratio = best / float(N)
        mean_occ = count / float(N)
        full = count >= N - 1
        half = count >= N // 2
        return dict(count=count, run_ratio=run_ratio, mean_occ=mean_occ, full=full, half=half)

    S = {k: stats(v) for k, v in sides.items()}
    adj_pairs = [("top", "right"), ("right", "bottom"), ("bottom", "left"), ("left", "top")]

    # rule 1: both full-ish
    full_pairs = [(a, b) for (a, b) in adj_pairs if S[a]["full"] and S[b]["full"]]
    if full_pairs:
        return max(full_pairs, key=lambda p: S[p[0]]["count"] + S[p[1]]["count"])

    # rule 2: both half-ish
    half_pairs = [(a, b) for (a, b) in adj_pairs if S[a]["half"] and S[b]["half"]]
    if half_pairs:
        return max(half_pairs, key=lambda p: S[p[0]]["count"] + S[p[1]]["count"])

    # fallback: run + occupancy
    def score_side(name):
        st = S[name]
        return 1.5 * st["run_ratio"] + 1.0 * st["mean_occ"] + 0.01 * st["count"]

    best_pair, best_score = None, -1e9
    for a, b in adj_pairs:
        sc = score_side(a) + score_side(b)
        if sc > best_score:
            best_score = sc
            best_pair = (a, b)
    return best_pair


def enforce_L_and_timing_by_sides(grid: np.ndarray, L_pair):
    """
    Force:
      - L sides ALL 1
      - timing sides strict 1010.. starting with black next to L
      - timing-timing corner ALWAYS 0 (white)
    """
    N = grid.shape[0]
    g = grid.copy()
    Lset = set(L_pair)

    def set_side(name, arr):
        if name == "top":
            g[0, :] = arr
        elif name == "bottom":
            g[-1, :] = arr
        elif name == "left":
            g[:, 0] = arr
        elif name == "right":
            g[:, -1] = arr

    def set_side_const(name, val):
        set_side(name, np.full(N, val, dtype=np.uint8))

    # L sides full
    for s in Lset:
        set_side_const(s, 1)

    # timing pattern depends on which corner is timing-timing
    if Lset == {"left", "bottom"}:
        # timing: top, right ; timing corner = TR must be 0
        top = np.array([(j % 2 == 0) for j in range(N)], dtype=np.uint8)  # 1,0,1,0.. => TR=0 for even N
        right = np.array([(i % 2 == 1) for i in range(N)], dtype=np.uint8)  # 0,1,0,1.. => TR=0
        set_side("top", top)
        set_side("right", right)

    elif Lset == {"bottom", "right"}:
        # timing: top, left ; timing corner = TL must be 0
        top = np.array([(j % 2 == 1) for j in range(N)], dtype=np.uint8)   # 0,1,0,1.. => TL=0
        left = np.array([(i % 2 == 0) for i in range(N)], dtype=np.uint8)  # 0,1,0,1.. => TL=0
        set_side("top", top)
        set_side("left", left)

    elif Lset == {"right", "top"}:
        # timing: bottom, left ; timing corner = BL must be 0
        bottom = np.array([(j % 2 == 1) for j in range(N)], dtype=np.uint8)  # 0,1,0,1.. => BL=0
        left = np.array([(i % 2 == 1) for i in range(N)], dtype=np.uint8)    # 1,0,1,0.. but we need BL=0 -> start with 1 at TL, BL=0
        # left currently makes TL=0 if i%2==0; we want TL=1 here (connected to L), BL=0 ok -> use i%2==0? no.
        # For (right,top) L corner is TR, so TL is timing+L? Actually TL is timing+? Top is L, left is timing. TL touches top(L), so TL should be 1.
        left = np.array([(i % 2 == 0) for i in range(N)], dtype=np.uint8)    # TL=1, BL=0
        set_side("bottom", bottom)
        set_side("left", left)

    elif Lset == {"top", "left"}:
        # timing: bottom, right ; timing corner = BR must be 0
        bottom = np.array([(j % 2 == 0) for j in range(N)], dtype=np.uint8)  # 1,0.. => BL=1, BR=0
        right = np.array([(i % 2 == 0) for i in range(N)], dtype=np.uint8)   # 1,0.. => TR=1, BR=0
        set_side("bottom", bottom)
        set_side("right", right)

    return g


def timing_corner_for_Lpair(L_pair):
    """Return which corner is timing-timing (allowed empty) given L_pair."""
    Lset = set(L_pair)
    timing = {"top", "right", "bottom", "left"} - Lset
    timing = tuple(sorted(list(timing)))
    # timing corner = intersection of timing sides
    if set(timing) == {"top", "right"}:
        return (0, -1)     # TR
    if set(timing) == {"right", "bottom"}:
        return (-1, -1)    # BR
    if set(timing) == {"bottom", "left"}:
        return (-1, 0)     # BL
    if set(timing) == {"left", "top"}:
        return (0, 0)      # TL
    # should not happen
    return (0, 0)


# ============================================================
# Step 4: L-anchored regrid (only when needed)
# ============================================================

def ransac_line(points, iters=500, thresh=2.5):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return None, None

    best_model = None
    best_inliers = []

    for _ in range(iters):
        i1, i2 = np.random.randint(0, len(pts), 2)
        if i1 == i2:
            continue
        x1, y1 = pts[i1]
        x2, y2 = pts[i2]
        if np.hypot(x2 - x1, y2 - y1) < 1e-3:
            continue

        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        denom = np.hypot(a, b)
        if denom < 1e-8:
            continue

        d = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / denom
        inliers = np.where(d < thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (a, b, c)

    return best_model, best_inliers


def intersect_lines(L1, L2):
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b1 * (-c2) - b2 * (-c1)) / det
    y = (a2 * (-c1) - a1 * (-c2)) / det
    return np.array([x, y], dtype=np.float32)


def line_direction(L):
    """Return a unit direction vector along the line."""
    a, b, _ = L
    # line normal is (a,b) => direction is (-b, a)
    t = np.array([-b, a], dtype=np.float32)
    n = np.linalg.norm(t)
    if n < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return t / n


def choose_sign_toward_center(origin, t, center):
    """Pick +t or -t to point toward image center."""
    v = center - origin
    return t if float(np.dot(v, t)) >= 0 else -t


def project_along(vectors, origin, dir_unit):
    """Project points onto dir."""
    rel = vectors - origin[None, :]
    return rel @ dir_unit


def robust_pitch_from_points(points_xy, origin, dir_unit, min_needed=5):
    """Median spacing of sorted projections."""
    if len(points_xy) < min_needed:
        return None
    pts = np.asarray(points_xy, dtype=np.float32)
    s = np.sort(project_along(pts, origin, dir_unit))
    if len(s) < min_needed:
        return None
    ds = np.diff(s)
    ds = ds[ds > 1e-3]
    if ds.size < 3:
        return None
    return float(np.median(ds))


def regrid_from_predicted_L(blobs, img_shape, N, L_pair):
    """
    Build a perfect rectangular grid anchored on the predicted L.
    Borders are assumed perfect; blobs are used ONLY for interior occupancy.
    """
    h, w = img_shape
    grid = np.zeros((N, N), dtype=np.uint8)

    # cell size from image bounds
    cell_x = w / float(N)
    cell_y = h / float(N)

    # precompute blob centers
    pts = np.stack([blobs[:, 1], blobs[:, 0]], axis=1) if len(blobs) else np.zeros((0,2))

    # fill interior cells only
    for i in range(N):
        for j in range(N):
            # skip borders (they will be forced later)
            if i == 0 or i == N-1 or j == 0 or j == N-1:
                continue

            cx = (j + 0.5) * cell_x
            cy = (i + 0.5) * cell_y

            if len(pts):
                d = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
                if np.min(d) < 0.45 * min(cell_x, cell_y):
                    grid[i, j] = 1

    return grid

def draw_parallelogram_grid(vis: np.ndarray, inter: np.ndarray, color=(0, 0, 255), thickness=1):
    """Draw cell boundaries from intersections."""
    Np1 = inter.shape[0]
    # horizontal segments
    for i in range(Np1):
        for j in range(Np1 - 1):
            p1 = inter[i, j]
            p2 = inter[i, j + 1]
            cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    # vertical segments
    for j in range(Np1):
        for i in range(Np1 - 1):
            p1 = inter[i, j]
            p2 = inter[i + 1, j]
            cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)


# ============================================================
# Output
# ============================================================

def grid_to_image(grid: np.ndarray, scale: int = 12) -> np.ndarray:
    # grid: 1=black module, 0=white module
    # image: 0=black, 255=white
    small = (1 - grid.astype(np.uint8)) * 255
    N = small.shape[0]
    return cv2.resize(small, (N * scale, N * scale), interpolation=cv2.INTER_NEAREST)


# ============================================================
# Laser
# ============================================================

def process_laser(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    blur = cv2.medianBlur(gray, 3)
    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 5
    )
    return thr


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True, help="Folder with rectified crops")
    ap.add_argument("--labels", required=True, help="Folder with label txts")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log_threshold", type=float, default=0.01, help="blob_log threshold (lower = more blobs)")
    args = ap.parse_args()

    img_dir = Path(args.imgs)
    lab_dir = Path(args.labels)
    out_dir = Path(args.out)
    dbg_dir = out_dir / "_debug"

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    img_paths = []
    for e in exts:
        img_paths += list(img_dir.glob(e))
    img_paths = sorted(img_paths)

    for img_path in img_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if gray0.dtype != np.uint8:
            gray0 = np.clip(gray0, 0, 255).astype(np.uint8)
        h, w = gray0.shape

        label_path = find_label_for_image(img_path, lab_dir)
        top = read_top_label(label_path)
        if top is None:
            continue
        cls_id, conf = top
        mode = "dotpeen" if cls_id == 0 else "laser" if cls_id == 1 else "unknown"
        print(f"[INFO] {img_path.name}: {mode} (conf={conf:.3f})")

        if mode == "laser":
            thr = process_laser(gray0)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_laser.png"), thr)
            if args.debug:
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_laser_debug.png"), thr)
            continue

        if mode != "dotpeen":
            continue

        # STEP 0
        gray_pol, hists = quadrant_polarity_fix(gray0)
        if args.debug:
            save_hist_panel(hists, dbg_dir / f"{img_path.stem}_step0_hists.png")
            cv2.imwrite(str(dbg_dir / f"{img_path.stem}_step0_polarity.png"), gray_pol)

        # STEP 1: blobs (on polarity image)
        blobs, gray_used, cell_virtual = detect_dots_log(
            gray_pol, grid_size_virtual=15, threshold=args.log_threshold
        )
        if len(blobs) == 0:
            print(f"[WARN] no blobs: {img_path.name}")
            continue

        # Run both N=14 and N=16
        for N in (14, 16):
            # Debug step1: raw blobs blue + red cell boundaries
            if args.debug:
                vis1 = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
                draw_cell_grid(vis1, N, color=(0, 0, 255), thickness=1)     # red
                draw_blobs(vis1, blobs, color_bgr=(255, 0, 0), thickness=2) # blue
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_N{N}_step1_raw_grid_blobs.png"), vis1)

            # STEP 2: filter blobs per cell
            blobs_f = filter_blobs_in_cells(blobs, gray_pol, N)

            if args.debug:
                vis2 = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
                draw_cell_grid(vis2, N, color=(0, 0, 255), thickness=1)      # red
                draw_blobs(vis2, blobs_f, color_bgr=(0, 255, 0), thickness=2) # green
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_N{N}_step2_filtered_grid_blobs.png"), vis2)

            grid_f = grid_from_blobs_rect(blobs_f, N, h, w)

            # STEP 3: choose L on filtered grid (before forcing)
            L_pair = choose_L_sides_strict(grid_f)

            # corner validation: allow ONLY timing-timing corner to be empty
            timing_corner = timing_corner_for_Lpair(L_pair)

            corners = {
                "TL": (0, 0),
                "TR": (0, N - 1),
                "BL": (N - 1, 0),
                "BR": (N - 1, N - 1),
            }
            # allowed empty corner:
            if timing_corner == (0, 0):
                allowed = "TL"
            elif timing_corner == (0, -1):
                allowed = "TR"
            elif timing_corner == (-1, 0):
                allowed = "BL"
            else:
                allowed = "BR"

            bad = False
            for name, (i, j) in corners.items():
                if name == allowed:
                    continue
                if grid_f[i, j] == 0:
                    bad = True
                    break

            # STEP 4: recalibrate grid if corner test fails
            inter = None
            if bad:  # corner validation failed
                grid_f = regrid_from_predicted_L(blobs_f, (h, w), N, L_pair)

                if args.debug:
                    vis3 = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)

                    # draw the (new) grid cell boundaries in red
                    draw_cell_grid(vis3, N, color=(0, 255, 255), thickness=1)

                    # draw blobs in green
                    draw_blobs(vis3, blobs_f, color_bgr=(0, 255, 0), thickness=2)

                    cv2.imwrite(
                        str(dbg_dir / f"{img_path.stem}_N{N}_step3_regrid.png"),
                        vis3
                    )


            # STEP 5: enforce border strictly
            grid_final = enforce_L_and_timing_by_sides(grid_f, L_pair)

            # STEP 6: synthetic
            syn = grid_to_image(grid_final, scale=12)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_N{N}_synthetic.png"), syn)

    print("[OK] Done.")


if __name__ == "__main__":
    main()
