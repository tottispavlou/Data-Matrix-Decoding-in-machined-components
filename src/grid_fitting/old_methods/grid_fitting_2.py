# border_iter_grid_fitting_locked_v2.py
#
# FULL pipeline (DOT-PEEN + LASER) with the latest decisions:
#  - Quadrant polarity correction + histogram debug
#  - LoG blob detection (your updated sigma settings)
#  - STRICT blob radius filter (reduce merged dots)
#  - Iterative border discovery with:
#       * per-side ACTIVE/STOP flags (depth cap is PER-SIDE, not global)
#       * LOCK = stop searching that side immediately
#       * debug shows ALL iterations + ALL tries to lock (TRY/LOCK/STOP)
#       * if some sides never lock but an adjacent pair locks (an L), we INFER the missing opposite sides
#         using parallel-line estimation (robust median projection) within the allowed border band.
#  - Grid creation from 4 sides -> bilinear (parallelogram) cell grid
#  - Cell filtering (spill >40% reject + keep best per cell by r*contrast)
#  - L + timing border forcing (L full, timing strict wbwb, timing-timing corner always white)
#  - Synthetic NxN output (N=16 first, then N=14)
#
# Debug outputs (--debug):
#   step0: *_step0_hists.png, *_step0_polarity.png
#   per N:
#     *_N{N}_iterXX.png            (each border iteration: locked + tries + status)
#     *_N{N}_final_grid_all.png    (final grid + all blobs)
#     *_N{N}_final_grid_kept.png   (final grid + kept blobs)
#
# Output:
#   *_N{N}_synthetic.png
# Laser output:
#   *_laser.png

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
            if not l:
                continue
            rows.append(l.split())
    if not rows:
        return None
    pairs = [(int(float(p[0])), float(p[-1])) for p in rows]
    return max(pairs, key=lambda x: x[1])


# ============================================================
# Step 0: Quadrant polarity + histogram debug
# ============================================================

def quadrant_polarity_fix(gray: np.ndarray):
    """
    Candidate inversion per quadrant if hist max peak is bright (>128).
    Invert ONLY if candidate has an adjacent candidate (TL-TR, TL-BL, TR-BR, BL-BR).
    """
    h, w = gray.shape
    hh, ww = h // 2, w // 2
    Q = [
        gray[0:hh, 0:ww],   # TL
        gray[0:hh, ww:w],   # TR
        gray[hh:h, 0:ww],   # BL
        gray[hh:h, ww:w],   # BR
    ]

    cand = [False] * 4
    hists = []
    for i, q in enumerate(Q):
        hist = cv2.calcHist([q], [0], None, [256], [0, 256]).flatten()
        hists.append(hist)
        peak = int(np.argmax(hist))
        cand[i] = (peak > 128)

    adj_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
    invert = [False] * 4
    for a, b in adj_pairs:
        if cand[a] and cand[b]:
            invert[a] = True
            invert[b] = True

    if not any(invert):
        invert = [False] * 4

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
# Step 1: LoG blobs (your updated settings) + strict radius filter
# ============================================================

def detect_dots_log(gray: np.ndarray, grid_size_virtual: int, threshold: float):
    """
    LoG with your settings:
      dot_radius = 0.5*cell
      sigma_est = dot_radius / sqrt(2)
      sigma_min = max(0.6, 0.65*sigma_est)
      sigma_max = 1.1*sigma_est
      num_sigma = 13
    Then strict radius filter:
      r in [0.30*cell, 0.55*cell] to reduce merged dots.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
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
    if blobs.size > 0:
        blobs[:, 2] *= np.sqrt(2.0)  # sigma -> radius

    r_min = 0.30 * cell
    r_max = 0.55 * cell
    keep = [b for b in blobs if r_min <= b[2] <= r_max]
    if not keep:
        return np.zeros((0, 3), dtype=np.float32), gray, cell

    return np.array(keep, dtype=np.float32), gray, cell


# ============================================================
# Geometry helpers
# ============================================================

def ransac_line(points_xy: np.ndarray, iters=650, dist_thresh=2.5):
    """
    Fit ax + by + c = 0 via RANSAC.
    points_xy: Nx2 float [x,y]
    Returns (a,b,c), inlier_idx
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) < 2:
        return None, None

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
            best_model = (a, b, c)

    return best_model, best_inliers


def line_dir(line_abc):
    a, b, _ = line_abc
    t = np.array([-b, a], dtype=np.float32)
    n = np.linalg.norm(t)
    if n < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return t / n


def intersect_lines(L1, L2):
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b1 * (-c2) - b2 * (-c1)) / det
    y = (a2 * (-c1) - a1 * (-c2)) / det
    return np.array([x, y], dtype=np.float32)


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


def draw_parallelogram_grid(vis, inter, color=(0, 0, 255), thickness=1):
    Np1 = inter.shape[0]
    for i in range(Np1):
        for j in range(Np1 - 1):
            p1 = inter[i, j]
            p2 = inter[i, j + 1]
            cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    for j in range(Np1):
        for i in range(Np1 - 1):
            p1 = inter[i, j]
            p2 = inter[i + 1, j]
            cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)


# ============================================================
# Border search bands + parallel inference (UNCHANGED)
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


def parallel_line_from_points(points_xy, dir_ref):
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) < 2:
        return None
    d = np.asarray(dir_ref, dtype=np.float32)
    d = d / (np.linalg.norm(d) + 1e-8)
    n = np.array([-d[1], d[0]], dtype=np.float32)  # perpendicular
    proj = pts @ n
    c = -float(np.median(proj))
    a, b = float(n[0]), float(n[1])
    return (a, b, c)


def count_inliers_for_line(points_xy, line_abc, dist_thresh):
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) == 0:
        return 0
    a, b, c = line_abc
    denom = np.hypot(a, b) + 1e-8
    d = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / denom
    return int(np.sum(d < dist_thresh))


def build_quad_from_4lines(locked):
    TL = intersect_lines(locked["top"], locked["left"])
    TR = intersect_lines(locked["top"], locked["right"])
    BR = intersect_lines(locked["bottom"], locked["right"])
    BL = intersect_lines(locked["bottom"], locked["left"])
    if any(p is None for p in (TL, TR, BR, BL)):
        return None
    return np.stack([TL, TR, BR, BL], axis=0)


def bilinear_intersections(quad, N):
    TL, TR, BR, BL = quad
    inter = np.zeros((N + 1, N + 1, 2), dtype=np.float32)
    for i in range(N + 1):
        v = i / float(N)
        left = TL * (1 - v) + BL * v
        right = TR * (1 - v) + BR * v
        for j in range(N + 1):
            u = j / float(N)
            inter[i, j] = left * (1 - u) + right * u
    return inter


def select_outermost_points(pts, side, k):
    if len(pts) <= k:
        return pts
    if side == "top":
        return pts[np.argsort(pts[:, 1])[:k]]        # smallest y
    if side == "bottom":
        return pts[np.argsort(pts[:, 1])[-k:]]       # largest y
    if side == "left":
        return pts[np.argsort(pts[:, 0])[:k]]        # smallest x
    if side == "right":
        return pts[np.argsort(pts[:, 0])[-k:]]       # largest x
    return pts


def try_find_border_iterative_locked(
    blobs, gray_used, N,
    dbg_dir=None, stem=None,
    max_iters=18,
    max_depth_cells=1.25
):
    """
    UNCHANGED from your version, including debug overlay text.
    Tight regulation is kept by using default max_depth_cells=1.25 (tighter than 1.5).
    """
    from pathlib import Path

    h, w = gray_used.shape
    if len(blobs) < 12:
        return None

    blobs_xy = np.stack([blobs[:, 1], blobs[:, 0]], axis=1).astype(np.float32)

    cellN = min(h, w) / float(N)
    band_step = 0.5 * cellN
    band_width = 1.05 * cellN
    dist_thresh = 0.35 * cellN

    min_L = N - 3
    min_timing = (N // 2) - 1

    sides = ("top", "right", "bottom", "left")
    opp = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

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

    def in_border_band(points_xy, side, depth):
        return side_band_points(points_xy, h, w, side, depth, band_width)

    def lock_side(side, L, cnt, depth):
        locked[side] = L
        locked_counts[side] = int(cnt)
        locked_depth[side] = float(depth)
        active[side] = False

    def find_any_adjacent_pair():
        adj = [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
        for a, b in adj:
            if locked[a] is not None and locked[b] is not None:
                return (a, b)
        return None

    def infer_side_parallel(target_side, ref_side):
        pts = in_border_band(blobs_xy, target_side, 0.0)
        if len(pts) < 6:
            return None, 0
        dir_ref = line_dir(locked[ref_side])
        L = parallel_line_from_points(pts, dir_ref)
        if L is None:
            return None, 0
        cnt = count_inliers_for_line(pts, L, dist_thresh)
        return L, cnt

    def clamp_and_validate_quad(quad):
        quad = quad.copy()
        quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
        return quad

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

            pts = in_border_band(blobs_xy, s, depth)
            if len(pts) < min_timing:
                last_try[s] = {"L": None, "cnt": 0, "depth": float(depth)}
                continue

            K = max(min_L, min_timing)
            pts_outer = select_outermost_points(pts, s, K)

            L, inl = ransac_line(pts_outer, iters=650, dist_thresh=dist_thresh)
            cnt = int(len(inl)) if inl is not None else 0
            last_try[s] = {"L": L, "cnt": cnt, "depth": float(depth)}

            if L is not None and can_lock(cnt):
                lock_side(s, L, cnt, depth)

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
            quad = clamp_and_validate_quad(quad)
            if quad is None:
                return None
            return {"quad": quad, "locked": locked, "counts": locked_counts,
                    "min_L": min_L, "min_timing": min_timing}

        if all(not active[s] for s in sides):
            break

    Lpair = find_any_adjacent_pair()
    if Lpair is None:
        return None

    for a, b in [("top", "bottom"), ("left", "right")]:
        if locked[a] is None and locked[b] is None:
            return None

        if locked[a] is None:
            L, cnt = infer_side_parallel(a, b)
            if L is None or cnt < max(4, min_timing - 2):
                return None
            lock_side(a, L, cnt, 0.0)

        if locked[b] is None:
            L, cnt = infer_side_parallel(b, a)
            if L is None or cnt < max(4, min_timing - 2):
                return None
            lock_side(b, L, cnt, 0.0)

    if not all(locked[s] is not None for s in sides):
        return None

    quad = build_quad_from_4lines(locked)
    if quad is None:
        return None
    quad = clamp_and_validate_quad(quad)
    if quad is None:
        return None

    return {"quad": quad, "locked": locked, "counts": locked_counts,
            "min_L": min_L, "min_timing": min_timing}


# ============================================================
# Cell mapping + filtering (CHANGED: polygon containment)
# ============================================================

def cell_polygon_from_inter(inter, i, j):
    TL = inter[i, j]
    TR = inter[i, j + 1]
    BR = inter[i + 1, j + 1]
    BL = inter[i + 1, j]
    return np.stack([TL, TR, BR, BL], axis=0)


def mean_in_circle(gray, cx, cy, r):
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


def blob_contrast(gray, cx, cy, r):
    m_in = mean_in_circle(gray, cx, cy, max(1.0, 0.6 * r))
    m_r1 = mean_in_circle(gray, cx, cy, max(1.0, 1.8 * r))
    m_r0 = mean_in_circle(gray, cx, cy, max(1.0, 1.2 * r))
    m_ring = max(0.0, m_r1 - m_r0)
    return abs(m_in - m_ring)


def spill_fraction_sample(cx, cy, r, cell_poly, samples=48):
    cnt = cell_poly.astype(np.float32).reshape(-1, 1, 2)
    outside = 0
    total = 0
    radii = [0.5 * r, 0.8 * r, 1.0 * r]
    per_ring = max(8, samples // len(radii))

    for rr in radii:
        for k in range(per_ring):
            ang = 2.0 * np.pi * (k / float(per_ring))
            px = cx + rr * np.cos(ang)
            py = cy + rr * np.sin(ang)
            total += 1
            inside = cv2.pointPolygonTest(cnt, (float(px), float(py)), False)
            if inside < 0:
                outside += 1

    return outside / float(max(1, total))

def uv_from_parallelogram(p, TL, TR, BL):
    """
    Solve p = TL + u*(TR-TL) + v*(BL-TL) for (u,v).
    Works for skewed (non-perpendicular) borders.
    """
    vx = TR - TL
    vy = BL - TL
    A = np.array([[vx[0], vy[0]],
                  [vx[1], vy[1]]], dtype=np.float32)

    det = float(A[0,0]*A[1,1] - A[0,1]*A[1,0])
    if abs(det) < 1e-6:
        return None  # degenerate

    b = (p - TL).astype(np.float32)
    # inv(A) * b for 2x2
    u = ( b[0]*A[1,1] - b[1]*A[0,1]) / det
    v = (-b[0]*A[1,0] + b[1]*A[0,0]) / det
    return float(u), float(v)


def map_blobs_to_grid_uv(blobs, quad, N, eps=0.08):
    """
    Map blobs to an NxN grid using proper (u,v) from a skewed parallelogram.
    eps allows slight outside tolerance (important for line-fit bias).
    """
    TL, TR, BR, BL = quad  # assumes your quad is in this order
    grid = np.zeros((N, N), dtype=np.uint8)

    mapped = 0
    for (y, x, r) in blobs:
        p = np.array([x, y], dtype=np.float32)
        uv = uv_from_parallelogram(p, TL, TR, BL)
        if uv is None:
            continue
        u, v = uv

        if not (-eps <= u <= 1.0 + eps and -eps <= v <= 1.0 + eps):
            continue

        u = min(1.0 - 1e-9, max(0.0, u))
        v = min(1.0 - 1e-9, max(0.0, v))

        j = int(u * N)
        i = int(v * N)
        i = max(0, min(N - 1, i))
        j = max(0, min(N - 1, j))

        grid[i, j] = 1
        mapped += 1

    print(f"[DBG] mapped blobs (uv-linear): {mapped}, filled cells: {int(grid.sum())}")
    return grid

def blobs_kept_from_uv(blobs, quad, N, eps=0.08):
    TL, TR, BR, BL = quad
    cell_rep = {}  # (i,j)->blob index

    for bi, (y, x, r) in enumerate(blobs):
        p = np.array([x, y], dtype=np.float32)
        uv = uv_from_parallelogram(p, TL, TR, BL)
        if uv is None:
            continue
        u, v = uv

        if not (-eps <= u <= 1.0 + eps and -eps <= v <= 1.0 + eps):
            continue

        u = min(1.0 - 1e-9, max(0.0, u))
        v = min(1.0 - 1e-9, max(0.0, v))

        j = int(u * N)
        i = int(v * N)
        i = max(0, min(N - 1, i))
        j = max(0, min(N - 1, j))

        if (i, j) not in cell_rep:
            cell_rep[(i, j)] = bi

    if not cell_rep:
        return np.zeros((0, 3), dtype=np.float32)

    kept_idx = sorted(cell_rep.values())
    return blobs[kept_idx]


def map_blobs_to_cells_polygon(blobs, inter):
    """
    Robust assignment: blob center must lie inside the cell polygon.
    Returns per_cell dict (i,j)->[blob indices].
    """
    N = inter.shape[0] - 1
    per_cell = {(i, j): [] for i in range(N) for j in range(N)}

    # Precompute polygons (tiny speed win)
    polys = [[cell_polygon_from_inter(inter, i, j).astype(np.float32).reshape(-1, 1, 2)
              for j in range(N)] for i in range(N)]

    mapped = 0
    for bi, (y, x, r) in enumerate(blobs):
        p = (float(x), float(y))
        found = False
        for i in range(N):
            for j in range(N):
                if cv2.pointPolygonTest(polys[i][j], p, False) >= 0:
                    per_cell[(i, j)].append(bi)
                    mapped += 1
                    found = True
                    break
            if found:
                break

    print(f"[DBG] mapped blobs: {mapped}")
    return per_cell


def grid_from_kept(blobs, kept_idx, inter):
    """
    After keeping at most one blob per cell, mark occupancy by polygon containment.
    """
    N = inter.shape[0] - 1
    grid = np.zeros((N, N), dtype=np.uint8)
    if kept_idx.size == 0:
        return grid

    kept = blobs[kept_idx]
    per_cell = map_blobs_to_cells_polygon(kept, inter)
    for (i, j), idxs in per_cell.items():
        if idxs:
            grid[i, j] = 1
    return grid

def order_quad_variants(quad):
    """
    Generate 8 variants of quad ordering:
    4 rotations of CCW order + 4 rotations of flipped order.
    Each variant is returned as (TL,TR,BR,BL).
    """
    q = quad.astype(np.float32)

    # 1) Order points cyclically around centroid (CCW)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q_ccw = q[np.argsort(ang)]

    variants = []

    def as_TLTRBRBL(qq):
        # qq is CCW: [p0,p1,p2,p3]
        # find TL as min(x+y)
        s = qq[:, 0] + qq[:, 1]
        k = np.argmin(s)
        qq = np.roll(qq, -k, axis=0)
        return np.stack([qq[0], qq[1], qq[2], qq[3]], axis=0)

    # 4 rotations CCW
    for k in range(4):
        variants.append(as_TLTRBRBL(np.roll(q_ccw, -k, axis=0)))

    # 4 rotations CW (flipped)
    q_cw = q_ccw[::-1].copy()
    for k in range(4):
        variants.append(as_TLTRBRBL(np.roll(q_cw, -k, axis=0)))

    return variants

def choose_best_quad_order(quad, blobs, N, eps=0.08):
    """
    Pick the quad ordering that maps the MOST blobs inside the unit square
    after homography (quad -> [0,1]^2).
    """
    pts = np.stack([blobs[:, 1], blobs[:, 0]], axis=1).astype(np.float32)  # [x,y]

    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    best = None
    best_in = -1
    best_H = None

    for q in order_quad_variants(quad):
        H = cv2.getPerspectiveTransform(q, dst)
        uv = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        u, v = uv[:, 0], uv[:, 1]
        inside = np.sum((u >= -eps) & (u <= 1 + eps) & (v >= -eps) & (v <= 1 + eps))
        if inside > best_in:
            best_in = int(inside)
            best = q
            best_H = H

    return best, best_H, best_in


def map_blobs_to_grid_homography(blobs, quad, N, eps=0.45):
    """
    Robust mapping: auto-fix quad order + homography map blobs to (u,v),
    then quantize to NxN.
    """
    q_best, H, inside = choose_best_quad_order(quad, blobs, N, eps=eps)

    pts = np.stack([blobs[:, 1], blobs[:, 0]], axis=1).astype(np.float32)
    uv = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)

    grid = np.zeros((N, N), dtype=np.uint8)
    mapped = 0
    for (u, v) in uv:
        if not (-eps <= u <= 1 + eps and -eps <= v <= 1 + eps):
            continue
        u = min(1.0 - 1e-9, max(0.0, float(u)))
        v = min(1.0 - 1e-9, max(0.0, float(v)))
        j = int(u * N)
        i = int(v * N)
        grid[i, j] = 1
        mapped += 1

    print(f"[DBG] quad_inside={inside}/{len(blobs)} mapped={mapped} filled={int(grid.sum())}")
    return grid, q_best


# ============================================================
# L selection (CHANGED: from border counts, not occupancy)
# ============================================================

def choose_L_sides_from_border_counts(border_counts):
    """
    Pick the adjacent pair with maximum total inliers.
    This is much more reliable than interior occupancy for dot-peen.
    """
    adj_pairs = [("top", "right"), ("right", "bottom"), ("bottom", "left"), ("left", "top")]
    def score(p):
        return int(border_counts.get(p[0], 0)) + int(border_counts.get(p[1], 0))
    return max(adj_pairs, key=score)


def enforce_L_and_timing_by_sides(grid, L_pair):
    """
    Force borders with strict DataMatrix rules:
      - L sides: all 1
      - timing sides: strict alternating
      - timing-timing corner: always 0 (white)
      - timing starts with black next to L (i.e., first timing cell adjacent to L corner is 1)
    """
    N = grid.shape[0]
    g = grid.copy()
    Lset = set(L_pair)
    all_sides = {"top", "right", "bottom", "left"}
    Tset = all_sides - Lset  # timing sides (2)

    def set_side(name, arr):
        if name == "top":
            g[0, :] = arr
        elif name == "bottom":
            g[-1, :] = arr
        elif name == "left":
            g[:, 0] = arr
        elif name == "right":
            g[:, -1] = arr

    # 1) L sides
    for s in Lset:
        set_side(s, np.ones(N, dtype=np.uint8))

    # 2) Determine the L-corner (intersection of L sides)
    # Corner indices: (row, col)
    if Lset == {"top", "left"}:
        Lcorner = (0, 0)
    elif Lset == {"top", "right"}:
        Lcorner = (0, N - 1)
    elif Lset == {"bottom", "left"}:
        Lcorner = (N - 1, 0)
    elif Lset == {"bottom", "right"}:
        Lcorner = (N - 1, N - 1)
    else:
        # fallback: do nothing fancy
        Lcorner = (0, 0)

    # 3) Timing sequences: start with black adjacent to Lcorner
    # Create base alternating pattern starting with 1 at index 0
    alt = np.array([(k % 2 == 0) for k in range(N)], dtype=np.uint8)

    # Apply to timing sides with orientation depending on which corner is L
    # The first timing cell adjacent to L corner should be black => alt[0]=1 at that adjacency.
    for s in Tset:
        if s in ("top", "bottom"):
            arr = alt.copy()
            # If Lcorner is on the right, reverse so adjacency at right has arr[-1]=1
            if Lcorner[1] == N - 1:
                arr = arr[::-1]
            set_side(s, arr)
        else:  # left/right
            arr = alt.copy()
            # If Lcorner is on the bottom, reverse so adjacency at bottom has arr[-1]=1
            if Lcorner[0] == N - 1:
                arr = arr[::-1]
            set_side(s, arr)

    # 4) timing-timing corner always white (0)
    # It's the corner NOT on any L side:
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
# Output (BULLETPROOF BINARIZED)
# ============================================================

def grid_to_image(grid, scale=12):
    # ensure 0/1
    g = (grid > 0).astype(np.uint8)
    small = (1 - g) * 255  # 1=black -> 0
    N = small.shape[0]
    out = cv2.resize(small, (N * scale, N * scale), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8)


# ============================================================
# Laser
# ============================================================

def process_laser(gray):
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
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log_threshold", type=float, default=0.01)
    ap.add_argument("--max_iters", type=int, default=18)
    ap.add_argument("--max_depth_cells", type=float, default=1.25)  # tighter regulation default
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

        if mode == "laser":
            thr = process_laser(gray0)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_laser.png"), thr)
            if args.debug:
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_laser_debug.png"), thr)
            continue

        if mode != "dotpeen":
            continue

        gray_pol, hists = quadrant_polarity_fix(gray0)
        if args.debug:
            save_hist_panel(hists, dbg_dir / f"{img_path.stem}_step0_hists.png")
            cv2.imwrite(str(dbg_dir / f"{img_path.stem}_step0_polarity.png"), gray_pol)

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

            quad = border["quad"]
            inter = bilinear_intersections(quad, N)

            if args.debug:
                vis_all = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
                draw_parallelogram_grid(vis_all, inter, (0, 0, 255), 1)
                draw_blobs(vis_all, blobs, (0, 255, 0), 2)
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_N{N}_final_grid_all.png"), vis_all)

            grid_occ, quad_ordered = map_blobs_to_grid_homography(blobs, quad, N)
            blobs_kept = blobs_kept_from_uv(blobs, quad, N)

            if args.debug:
                blobs_kept = blobs_kept_from_uv(blobs, quad, N)
                vis_kept = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
                draw_parallelogram_grid(vis_kept, inter, (0, 0, 255), 1)
                draw_blobs(vis_kept, blobs_kept, (0, 255, 0), 2)
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_N{N}_final_grid_kept.png"), vis_kept)


            # CHANGED: choose L from border counts, not interior occupancy
            L_pair = choose_L_sides_from_border_counts(border["counts"])

            # border forcing unchanged (includes timing-timing corner always white)
            grid_final = enforce_L_and_timing_by_sides(grid_occ, L_pair)

            syn = grid_to_image(grid_final, scale=12)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_N{N}_synthetic.png"), syn)

            solved = True
            break

        if not solved:
            print(f"[FAIL] border discovery failed for N=16 and N=14: {img_path.name}")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
