import argparse
from pathlib import Path
import numpy as np
import cv2
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
    rows = []
    with open(label_path, "r") as f:
        for l in f:
            l = l.strip()
            if l:
                rows.append(l.split())
    if not rows:
        return None
    pairs = [(int(float(p[0])), float(p[-1])) for p in rows]
    return max(pairs, key=lambda x: x[1])


# ============================================================
# Step 0: Quadrant polarity + histogram debug
# ============================================================

def quadrant_polarity_fix(gray: np.ndarray):
    h, w = gray.shape
    hh, ww = h // 2, w // 2
    Q = [
        gray[:hh, :ww],
        gray[:hh, ww:],
        gray[hh:, :ww],
        gray[hh:, ww:]
    ]

    cand = [False] * 4
    hists = []
    for i, q in enumerate(Q):
        hist = cv2.calcHist([q], [0], None, [256], [0, 256]).flatten()
        hists.append(hist)
        cand[i] = (int(np.argmax(hist)) > 128)

    # invert only when adjacent quadrants agree
    invert = [False] * 4
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        if cand[a] and cand[b]:
            invert[a] = True
            invert[b] = True

    fixed = []
    for i, q in enumerate(Q):
        fixed.append(255 - q if invert[i] else q.copy())

    out = np.vstack([np.hstack([fixed[0], fixed[1]]),
                     np.hstack([fixed[2], fixed[3]])])
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

def detect_dots_log(gray: np.ndarray, grid_size_virtual: int, threshold: float):
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    h, w = gray.shape
    cell = min(h, w) / float(grid_size_virtual)

    dot_radius = 0.5 * cell
    sigma_est = dot_radius / np.sqrt(2.0)
    sigma_min = max(0.6, 0.65 * sigma_est)
    sigma_max = 1.1 * sigma_est

    img_norm = gray.astype(np.float32) / 255.0
    blobs = blob_log(
        img_norm,
        min_sigma=sigma_min,
        max_sigma=sigma_max,
        num_sigma=13,
        threshold=threshold,
        overlap=0.1
    )
    if blobs.size:
        blobs[:, 2] *= np.sqrt(2.0)

    # strict radius filter
    r_min = 0.30 * cell
    r_max = 0.55 * cell
    blobs = np.array([b for b in blobs if r_min <= b[2] <= r_max], dtype=np.float32)
    return blobs, gray, cell


# ============================================================
# Laser
# ============================================================

def process_laser(gray: np.ndarray):
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
# Geometry helpers
# ============================================================

def ransac_line(points_xy: np.ndarray, iters=650, dist_thresh=2.5):
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) < 2:
        return None, np.array([], dtype=np.int32)

    best_model = None
    best_inliers = np.array([], dtype=np.int32)

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
        inliers = np.where(d < dist_thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (float(a), float(b), float(c))

    return best_model, best_inliers


def line_dir(L):
    a, b, _ = L
    d = np.array([-b, a], dtype=np.float32)
    n = np.linalg.norm(d)
    return d / (n + 1e-8)


def line_normal_unit(L):
    a, b, c = L
    n = np.hypot(a, b) + 1e-8
    return (a / n, b / n, c / n)


def intersect_lines(L1, L2):
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b1 * (-c2) - b2 * (-c1)) / det
    y = (a2 * (-c1) - a1 * (-c2)) / det
    return np.array([x, y], dtype=np.float32)


def line_from_point_dir(p, d):
    n = np.array([-d[1], d[0]], dtype=np.float32)
    a, b = float(n[0]), float(n[1])
    c = -a * float(p[0]) - b * float(p[1])
    return (a, b, c)


def draw_infinite_line(vis, L, color, thickness=2):
    h, w = vis.shape[:2]
    a, b, c = L
    if abs(b) > abs(a):
        x0, x1 = 0, w - 1
        y0 = int(round((-a * x0 - c) / (b + 1e-8)))
        y1 = int(round((-a * x1 - c) / (b + 1e-8)))
        cv2.line(vis, (x0, y0), (x1, y1), color, thickness)
    else:
        y0, y1 = 0, h - 1
        x0 = int(round((-b * y0 - c) / (a + 1e-8)))
        x1 = int(round((-b * y1 - c) / (a + 1e-8)))
        cv2.line(vis, (x0, y0), (x1, y1), color, thickness)


def draw_blobs(vis, blobs, color_bgr, thickness=2):
    for (y, x, r) in blobs:
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(r)), color_bgr, thickness, cv2.LINE_AA)


def draw_grid_from_centers(vis, centers, color=(0, 0, 255), thickness=1):
    """
    centers: (N,N,2) intersections at module centers.
    Draw a mesh by connecting adjacent centers.
    """
    N = centers.shape[0]
    for i in range(N):
        for j in range(N - 1):
            p1 = centers[i, j]
            p2 = centers[i, j + 1]
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    for j in range(N):
        for i in range(N - 1):
            p1 = centers[i, j]
            p2 = centers[i + 1, j]
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)


# ============================================================
# Border finder (kept compatible with your debug style)
# ============================================================

def side_band_points(blobs_xy, h, w, side, depth, band_width):
    x = blobs_xy[:, 0]
    y = blobs_xy[:, 1]
    if side == "top":
        m = y < (depth + band_width)
    elif side == "bottom":
        m = y > (h - (depth + band_width))
    elif side == "left":
        m = x < (depth + band_width)
    elif side == "right":
        m = x > (w - (depth + band_width))
    else:
        m = np.zeros((len(blobs_xy),), dtype=bool)
    return blobs_xy[m]


def select_outermost_points(pts, side, k):
    if len(pts) <= k:
        return pts
    if side == "top":
        return pts[np.argsort(pts[:, 1])[:k]]
    if side == "bottom":
        return pts[np.argsort(pts[:, 1])[-k:]]
    if side == "left":
        return pts[np.argsort(pts[:, 0])[:k]]
    if side == "right":
        return pts[np.argsort(pts[:, 0])[-k:]]
    return pts


def build_quad_from_4lines(locked):
    TL = intersect_lines(locked["top"], locked["left"])
    TR = intersect_lines(locked["top"], locked["right"])
    BR = intersect_lines(locked["bottom"], locked["right"])
    BL = intersect_lines(locked["bottom"], locked["left"])
    if any(p is None for p in (TL, TR, BR, BL)):
        return None
    return np.stack([TL, TR, BR, BL], axis=0)


def try_find_border_iterative_locked(
    blobs, gray_used, N,
    dbg_dir=None, stem=None,
    max_iters=18,
    max_depth_cells=1.25
):
    h, w = gray_used.shape
    if len(blobs) < 12:
        return None

    blobs_xy = np.stack([blobs[:, 1], blobs[:, 0]], axis=1).astype(np.float32)  # [x,y]

    cellN = min(h, w) / float(N)
    band_step = 0.5 * cellN
    band_width = 1.05 * cellN
    dist_thresh = 0.35 * cellN

    min_L = N - 3
    min_timing = (N // 2) - 1

    sides = ("top", "right", "bottom", "left")

    locked = {s: None for s in sides}
    locked_counts = {s: 0 for s in sides}
    locked_depth = {s: None for s in sides}
    active = {s: True for s in sides}
    last_try = {s: {"L": None, "cnt": 0, "depth": 0.0} for s in sides}

    def side_status(s):
        if locked[s] is not None:
            return "LOCK"
        if not active[s]:
            return "STOP"
        return "TRY"

    def can_lock(cnt):
        return cnt >= min_timing

    for it in range(max_iters):
        depth = it * band_step

        for s in sides:
            if locked[s] is not None:
                active[s] = False
            elif active[s] and (depth > max_depth_cells * cellN):
                active[s] = False

        for s in sides:
            if not active[s]:
                continue

            pts = side_band_points(blobs_xy, h, w, s, depth, band_width)
            if len(pts) < min_timing:
                last_try[s] = {"L": None, "cnt": 0, "depth": float(depth)}
                continue

            K = max(min_L, min_timing)
            pts_outer = select_outermost_points(pts, s, K)

            L, inl = ransac_line(pts_outer, iters=650, dist_thresh=dist_thresh)
            cnt = int(len(inl)) if inl is not None else 0
            last_try[s] = {"L": L, "cnt": cnt, "depth": float(depth)}

            if L is not None and can_lock(cnt):
                locked[s] = L
                locked_counts[s] = cnt
                locked_depth[s] = float(depth)
                active[s] = False

        # Debug iteration image (similar overlay text)
        if dbg_dir is not None and stem is not None:
            vis = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
            draw_blobs(vis, blobs, (255, 0, 0), 2)

            col_lock = (0, 255, 255)
            col_try = (255, 0, 255)

            for s in sides:
                if locked[s] is not None:
                    draw_infinite_line(vis, locked[s], col_lock, 2)

            for s in sides:
                if side_status(s) == "TRY":
                    L = last_try[s]["L"]
                    if L is not None:
                        draw_infinite_line(vis, L, col_try, 1)

            y0 = 22
            for k, s in enumerate(("top", "right", "bottom", "left")):
                st = side_status(s)
                if st == "LOCK":
                    msg = f"{s}:{st} cnt={locked_counts[s]} d={locked_depth[s]:.1f}"
                    col = col_lock
                elif st == "TRY":
                    msg = f"{s}:{st} cnt={last_try[s]['cnt']} d={last_try[s]['depth']:.1f}"
                    col = col_try
                else:
                    msg = f"{s}:{st}"
                    col = (170, 170, 170)
                cv2.putText(vis, msg, (10, y0 + 18 * k),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            cv2.putText(
                vis,
                f"N={N} it={it} depth={depth:.1f} minL={min_L} minT={min_timing} max_depth={max_depth_cells}cells",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA
            )
            cv2.imwrite(str(Path(dbg_dir) / f"{stem}_N{N}_iter{it:02d}.png"), vis)

        if all(locked[s] is not None for s in sides):
            quad = build_quad_from_4lines(locked)
            if quad is None:
                return None
            quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
            return {"quad": quad, "locked": locked, "counts": locked_counts,
                    "min_L": min_L, "min_timing": min_timing}

        if all(not active[s] for s in sides):
            break

    # If not all 4 locked, fail (keep it simple here)
    return None


# ============================================================
# Choose L sides + force borders (with correct corner + strict alternation)
# ============================================================

def choose_L_sides_from_border_counts(border_counts):
    adj_pairs = [("top", "right"), ("right", "bottom"), ("bottom", "left"), ("left", "top")]
    return max(adj_pairs, key=lambda p: int(border_counts.get(p[0], 0)) + int(border_counts.get(p[1], 0)))


def enforce_L_and_timing_by_sides(grid, L_pair):
    N = grid.shape[0]
    g = (grid > 0).astype(np.uint8).copy()

    Lset = set(L_pair)
    all_sides = {"top", "right", "bottom", "left"}
    Tset = all_sides - Lset

    def set_side(name, arr):
        if name == "top":
            g[0, :] = arr
        elif name == "bottom":
            g[-1, :] = arr
        elif name == "left":
            g[:, 0] = arr
        elif name == "right":
            g[:, -1] = arr

    # L sides full black
    for s in Lset:
        set_side(s, np.ones(N, dtype=np.uint8))

    # Determine L corner
    if Lset == {"top", "left"}:
        Lcorner = (0, 0)
    elif Lset == {"top", "right"}:
        Lcorner = (0, N - 1)
    elif Lset == {"bottom", "left"}:
        Lcorner = (N - 1, 0)
    else:  # {"bottom","right"}
        Lcorner = (N - 1, N - 1)

    # Alternation base (starts with black at index 0)
    alt = np.array([(k % 2 == 0) for k in range(N)], dtype=np.uint8)

    for s in Tset:
        if s in ("top", "bottom"):
            arr = alt.copy()
            if Lcorner[1] == N - 1:
                arr = arr[::-1]
            set_side(s, arr)
        else:
            arr = alt.copy()
            if Lcorner[0] == N - 1:
                arr = arr[::-1]
            set_side(s, arr)

    # timing-timing corner always white
    if Tset == {"top", "right"}:
        g[0, N - 1] = 0
    elif Tset == {"top", "left"}:
        g[0, 0] = 0
    elif Tset == {"bottom", "right"}:
        g[N - 1, N - 1] = 0
    elif Tset == {"bottom", "left"}:
        g[N - 1, 0] = 0

    return g


# ============================================================
# New approach: start from L sides, step by pitch, refit each row/col
# ============================================================

def orient_lines_interior_negative(locked_lines, interior_point_xy):
    """
    Flip each line so that interior_point satisfies ax+by+c <= 0
    Returns normalized (a,b,c) with unit normal.
    """
    out = {}
    x0, y0 = float(interior_point_xy[0]), float(interior_point_xy[1])
    for s, L in locked_lines.items():
        a, b, c = L
        val = a * x0 + b * y0 + c
        if val > 0:
            a, b, c = -a, -b, -c
        out[s] = line_normal_unit((a, b, c))
    return out


def filter_blobs_inside_halfplanes(blobs, borders_unit):
    """
    Keep blobs with ax+by+c <= 0 for all 4 sides (interior).
    borders_unit: dict top/bottom/left/right with unit-normal lines.
    """
    kept = []
    for (y, x, r) in blobs:
        xx, yy = float(x), float(y)
        ok = True
        for (a, b, c) in borders_unit.values():
            if a * xx + b * yy + c > 0:
                ok = False
                break
        if ok:
            kept.append((y, x, r))
    return np.array(kept, dtype=np.float32)


def signed_dist_to_line_unit(L_unit, x, y):
    a, b, c = L_unit
    return a * x + b * y + c  # since (a,b) is unit


def offset_line_unit(L_unit, delta_inward):
    """
    For unit-normal line ax+by+c=0 with interior being <=0,
    moving inward by delta corresponds to c' = c + delta.
    """
    a, b, c = L_unit
    return (a, b, c + float(delta_inward))


def fit_parallel_like(points_xy, base_dir, dist_thresh):
    """
    Fit a line through points, roughly parallel to base_dir.
    Easiest: do RANSAC then accept if direction not crazy.
    """
    L, inl = ransac_line(points_xy, iters=500, dist_thresh=dist_thresh)
    if L is None:
        return None, 0
    d = line_dir(L)
    bd = base_dir / (np.linalg.norm(base_dir) + 1e-8)
    if abs(float(np.dot(d, bd))) < 0.5:
        # too different; reject
        return None, 0
    return L, int(len(inl))


def build_rows_cols_from_L(
    blobs_in, borders_unit, locked_orig, N, L_pair, pitch_px,
    band_frac=0.45
):
    """
    Build N row lines and N col lines, starting from the two L sides.
    We step inward from each L side by k*pitch.
    For each step, we collect blobs near that offset line and fit a line.
    If fit fails, we fall back to the offset line.
    """
    # Decide which side controls rows and which controls cols
    Lset = set(L_pair)
    if Lset in ({"top", "left"}, {"top", "right"}):
        row_side = "top"
    else:
        row_side = "bottom"

    if Lset in ({"top", "left"}, {"bottom", "left"}):
        col_side = "left"
    else:
        col_side = "right"

    # base directions to keep row/col consistent
    dt = line_dir(locked_orig["top"])
    db = line_dir(locked_orig["bottom"])
    dl = line_dir(locked_orig["left"])
    dr = line_dir(locked_orig["right"])

    row_base_dir = dt if row_side == "top" else db
    col_base_dir = dl if col_side == "left" else dr

    band = band_frac * pitch_px
    dist_thresh_fit = 0.35 * pitch_px

    blobs_xy = np.stack([blobs_in[:, 1], blobs_in[:, 0]], axis=1).astype(np.float32)  # [x,y]

    rows = []
    cols = []
    row_used_mask = np.zeros((len(blobs_xy),), dtype=bool)
    col_used_mask = np.zeros((len(blobs_xy),), dtype=bool)

    # Build row lines
    base_row = borders_unit[row_side]
    for k in range(N):
        Lk = offset_line_unit(base_row, k * pitch_px)
        # select blobs near this line
        d = np.abs(signed_dist_to_line_unit(Lk, blobs_xy[:, 0], blobs_xy[:, 1]))
        idx = np.where(d <= band)[0]
        pts = blobs_xy[idx]
        Lfit, cnt = fit_parallel_like(pts, row_base_dir, dist_thresh_fit)
        if Lfit is None or cnt < max(4, N // 4):
            rows.append(Lk)  # fallback: use offset line
        else:
            # normalize fitted line to unit normal & same interior sign as row_side
            a, b, c = Lfit
            # ensure interior negative w.r.t. a point slightly inward from Lk (we can use image center)
            rows.append(line_normal_unit((a, b, c)))
            row_used_mask[idx] = True

    # Build col lines
    base_col = borders_unit[col_side]
    for k in range(N):
        Lk = offset_line_unit(base_col, k * pitch_px)
        d = np.abs(signed_dist_to_line_unit(Lk, blobs_xy[:, 0], blobs_xy[:, 1]))
        idx = np.where(d <= band)[0]
        pts = blobs_xy[idx]
        Lfit, cnt = fit_parallel_like(pts, col_base_dir, dist_thresh_fit)
        if Lfit is None or cnt < max(4, N // 4):
            cols.append(Lk)
        else:
            a, b, c = Lfit
            cols.append(line_normal_unit((a, b, c)))
            col_used_mask[idx] = True

    # "kept blobs" for debug = blobs that were used by at least one band selection
    used = row_used_mask | col_used_mask
    return rows, cols, used


def intersect_rows_cols(rows, cols):
    N = len(rows)
    centers = np.zeros((N, N, 2), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            p = intersect_lines(rows[i], cols[j])
            if p is None:
                p = np.array([np.nan, np.nan], dtype=np.float32)
            centers[i, j] = p
    return centers


def map_blobs_to_nearest_center(blobs_in, centers):
    """
    centers: (N,N,2) module centers
    For each blob, assign to nearest center => grid cell black.
    """
    N = centers.shape[0]
    grid = np.zeros((N, N), dtype=np.uint8)

    # flatten centers for speed
    C = centers.reshape(-1, 2)
    valid = ~np.any(np.isnan(C), axis=1)
    Cvalid = C[valid]
    if len(Cvalid) == 0:
        return grid

    for (y, x, r) in blobs_in:
        p = np.array([float(x), float(y)], dtype=np.float32)
        d = np.linalg.norm(Cvalid - p[None, :], axis=1)
        k = int(np.argmin(d))
        # map back to (i,j)
        idx_flat = np.where(valid)[0][k]
        i = idx_flat // N
        j = idx_flat % N
        grid[i, j] = 1

    return grid


# ============================================================
# Output: binarized + 2-module quiet zone padding
# ============================================================

def grid_to_image(grid, scale=12):
    g = (grid > 0).astype(np.uint8)
    img = (1 - g) * 255
    N = img.shape[0]
    out = cv2.resize(img, (N * scale, N * scale), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8)


def pad_modules(grid, pad=2):
    N = grid.shape[0]
    out = np.zeros((N + 2 * pad, N + 2 * pad), dtype=np.uint8)  # white modules
    out[pad:pad + N, pad:pad + N] = grid
    return out


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log_threshold", type=float, default=0.01)
    ap.add_argument("--max_iters", type=int, default=18)
    ap.add_argument("--max_depth_cells", type=float, default=1.25)
    ap.add_argument("--scale", type=int, default=12)
    ap.add_argument("--quiet", type=int, default=2)  # 2-module quiet zone
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

        label_path = find_label_for_image(img_path, lab_dir)
        top = read_top_label(label_path)
        if top is None:
            continue
        cls_id, conf = top
        mode = "dotpeen" if cls_id == 0 else "laser" if cls_id == 1 else "unknown"
        print(f"[INFO] {img_path.name}: {mode} (conf={conf:.3f})")

        # ---- laser ----
        if mode == "laser":
            thr = process_laser(gray0)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_laser.png"), thr)
            if args.debug:
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_laser_debug.png"), thr)
            continue

        if mode != "dotpeen":
            continue

        # ---- polarity fix ----
        gray_pol, hists = quadrant_polarity_fix(gray0)
        if args.debug:
            save_hist_panel(hists, dbg_dir / f"{img_path.stem}_step0_hists.png")
            cv2.imwrite(str(dbg_dir / f"{img_path.stem}_step0_polarity.png"), gray_pol)

        # ---- blobs ----
        blobs, gray_used, _ = detect_dots_log(gray_pol, grid_size_virtual=15, threshold=args.log_threshold)
        if len(blobs) == 0:
            print(f"[WARN] no blobs after LoG+r_filter: {img_path.name}")
            continue

        solved = False

        for N in (16, 14):
            border = try_find_border_iterative_locked(
                blobs=blobs,
                gray_used=gray_used,
                N=N,
                dbg_dir=(dbg_dir if args.debug else None),
                stem=img_path.stem,
                max_iters=args.max_iters,
                max_depth_cells=args.max_depth_cells
            )
            if border is None:
                continue

            locked_orig = border["locked"]
            quad = border["quad"]

            # interior reference point = quad center
            interior = quad.mean(axis=0)

            # orient border half-planes so interior is negative
            borders_unit = orient_lines_interior_negative(locked_orig, interior)

            # keep only blobs inside 4 half-planes
            blobs_in = filter_blobs_inside_halfplanes(blobs, borders_unit)
            if len(blobs_in) == 0:
                continue

            # decide L sides from border counts
            L_pair = choose_L_sides_from_border_counts(border["counts"])

            # pitch (module centers): distance between opposite borders / (N-1)
            # use mean absolute signed distance between the two lines:
            # since unit normal, distance between parallel lines is |c2 - c1|
            top_u = borders_unit["top"]
            bot_u = borders_unit["bottom"]
            left_u = borders_unit["left"]
            right_u = borders_unit["right"]

            # If these are not exactly parallel, this is still a stable scalar:
            pitch_rows = abs(bot_u[2] - top_u[2]) / float(N - 1)
            pitch_cols = abs(right_u[2] - left_u[2]) / float(N - 1)
            pitch = 0.5 * (pitch_rows + pitch_cols)

            # build rows & cols starting from L sides
            rows, cols, used_mask = build_rows_cols_from_L(
                blobs_in=blobs_in,
                borders_unit=borders_unit,
                locked_orig=locked_orig,
                N=N,
                L_pair=L_pair,
                pitch_px=pitch,
                band_frac=0.45
            )

            centers = intersect_rows_cols(rows, cols)

            # debug: grid + all blobs
            if args.debug:
                vis_all = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
                draw_grid_from_centers(vis_all, centers, (0, 0, 255), 1)
                draw_blobs(vis_all, blobs, (255, 0, 0), 2)
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_N{N}_final_grid_all.png"), vis_all)

            # map blobs to grid by nearest intersection
            grid_occ = map_blobs_to_nearest_center(blobs_in, centers)

            # debug: grid + kept blobs (those used in band selection)
            if args.debug:
                # blobs_in are already inside; show used subset as "kept"
                blobs_xy_in = np.stack([blobs_in[:, 1], blobs_in[:, 0]], axis=1)
                blobs_kept = blobs_in[used_mask] if len(used_mask) == len(blobs_in) else blobs_in

                vis_kept = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
                draw_grid_from_centers(vis_kept, centers, (0, 0, 255), 1)
                draw_blobs(vis_kept, blobs_kept, (0, 255, 0), 2)
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_N{N}_final_grid_kept.png"), vis_kept)

            # force borders (L + timing + timing corner white)
            grid_final = enforce_L_and_timing_by_sides(grid_occ, L_pair)

            # add 2-module quiet zone
            grid_out = pad_modules(grid_final, pad=args.quiet)

            syn = grid_to_image(grid_out, scale=args.scale)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_N{N}_synthetic.png"), syn)

            solved = True
            break

        if not solved:
            print(f"[FAIL] border discovery or grid build failed: {img_path.name}")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
