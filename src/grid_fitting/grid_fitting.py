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
# Quadrant polarity + histogram debug
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
        cand[i] = (int(np.argmax(hist)) > 130)

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
# Laser
# ============================================================

def process_laser(gray):
    """
    Contrast-only preprocessing for laser-marked DMCs.

    - Enhances dark/light separation
    - Preserves module shapes
    """

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )
    out = clahe.apply(gray)

    return out

def pad_image_quiet_zone(img, pad_modules, scale, value=255):
    """
    Pad an image with a quiet zone measured in modules.

    img          : uint8 image (grayscale or binary)
    pad_modules  : number of modules to pad (e.g. 2)
    scale        : pixels per module
    value        : background value (255 = white)
    """
    pad_px = int(pad_modules * scale)

    return cv2.copyMakeBorder(
        img,
        pad_px, pad_px, pad_px, pad_px,
        borderType=cv2.BORDER_CONSTANT,
        value=value
    )

# ============================================================
# LoG blobs (dot-peen)
# ============================================================

def detect_dots_log(gray: np.ndarray, grid_size_virtual: int, threshold: float):
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    h, w = gray.shape
    cell = min(h, w) / float(grid_size_virtual)

    dot_radius = 0.5 * cell
    sigma_est = dot_radius / np.sqrt(2.0)
    sigma_min = 0.75 * sigma_est
    sigma_max = sigma_est

    img_norm = gray.astype(np.float32) / 255.0
    blobs = blob_log(
        img_norm,
        min_sigma=sigma_min,
        max_sigma=sigma_max,
        num_sigma=8,
        threshold=threshold,
        overlap=0.05
    )
    if blobs.size:
        blobs[:, 2] *= np.sqrt(2.0)

    # strict radius filter to avoid merged / tiny dots
    r_min = 0.30 * cell
    r_max = 0.55 * cell
    blobs = np.array([b for b in blobs if r_min <= b[2] <= r_max], dtype=np.float32)
    return blobs, gray, cell

# ============================================================
# Drawing helpers
# ============================================================

def draw_blobs(vis, blobs, color_bgr, thickness=2):
    for (y, x, r) in blobs:
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(r)), color_bgr, thickness, cv2.LINE_AA)


def draw_grid_on_image(vis, N, color=(0, 0, 255), thickness=1):
    h, w = vis.shape[:2]
    for i in range(1, N):
        x = int(round(i * w / N))
        cv2.line(vis, (x, 0), (x, h - 1), color, thickness)
    for j in range(1, N):
        y = int(round(j * h / N))
        cv2.line(vis, (0, y), (w - 1, y), color, thickness)

def order_corners(pts4):
    """Same ordering as rectify_crops_segm.py: TL, TR, BR, BL."""
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_to_square_nearest(gray, corners, out_size=240):
    """Same warping style as rectify_crops_segm.py (nearest)."""
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(gray, M, (out_size, out_size), flags=cv2.INTER_NEAREST)


def crop_by_quad(gray, quad, pad=10):
    """
    Crop image tightly around quad bbox, and shift quad coords into crop space.
    This prevents black warps when quad extends slightly outside image.
    """
    q = np.asarray(quad, dtype=np.float32)
    h, w = gray.shape[:2]
    xs = q[:, 0]
    ys = q[:, 1]
    x0 = int(max(0, np.floor(xs.min()) - pad))
    y0 = int(max(0, np.floor(ys.min()) - pad))
    x1 = int(min(w, np.ceil(xs.max()) + pad))
    y1 = int(min(h, np.ceil(ys.max()) + pad))

    crop = gray[y0:y1, x0:x1].copy()
    q_shift = q - np.array([x0, y0], dtype=np.float32)
    return crop, q_shift, (x0, y0, x1, y1)


def intersect_abc(L1, L2):
    """Intersect two lines in ax+by+c=0 form."""
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (c1 * a2 - c2 * a1) / d
    return np.array([x, y], dtype=np.float32)


def quad_from_abc_lines(lines_abc):
    """
    Expect dict with keys: top,right,bottom,left each as (a,b,c).
    Returns quad TL,TR,BR,BL in pixel coords.
    """
    top = lines_abc["top"]
    right = lines_abc["right"]
    bottom = lines_abc["bottom"]
    left = lines_abc["left"]

    TL = intersect_abc(top, left)
    TR = intersect_abc(top, right)
    BR = intersect_abc(bottom, right)
    BL = intersect_abc(bottom, left)
    if TL is None or TR is None or BR is None or BL is None:
        return None
    return np.stack([TL, TR, BR, BL], axis=0).astype(np.float32)


def put_debug_text(vis, lines, xy=(10, 18), color=(230, 230, 230), scale=0.32):
    x, y = xy
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (x, y + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def estimate_pitch_only(blobs, min_points=20):
    """Estimate module pitch (pixels) from blob centers using NN median."""
    if len(blobs) < min_points:
        return None
    pts = np.stack([blobs[:, 1], blobs[:, 0]], axis=1).astype(np.float32)  # [x,y]
    pts = np.round(pts, 1)
    _, uniq_idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(uniq_idx)]
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    pitch = float(np.median(nn))
    if not np.isfinite(pitch) or pitch < 1.0:
        return None
    return pitch


# ============================================================
# Fast grid fit: lay N×N over whole image
# ============================================================

def fast_grid_fit_whole_image(blobs_xy, shape_hw, N):
    """
    Lay an NxN grid over the whole image.
    """
    h, w = shape_hw
    cw = w / N
    ch = h / N

    grid = np.zeros((N, N), np.uint8)

    for x, y in blobs_xy:
        i = int(x / cw)
        j = int(y / ch)
        if 0 <= i < N and 0 <= j < N:
            grid[j, i] = 1

    return grid

# ============================================================
# Border forcing for synthetic grid
# ============================================================

def enforce_L_and_timing_by_sides(grid, L_pair, timing_corner_value=0):
    """
    grid is N×N occupancy. Force:
      - L sides fully black
      - timing sides alternating
      - timing-timing corner forced (user requested: treat as black -> 1)
    """
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
    else:
        Lcorner = (N - 1, N - 1)

    # Alternation base
    alt = np.array([(k % 2 == 0) for k in range(N)], dtype=np.uint8)

    # set timing sides
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

    # timing-timing corner forced (per user's latest note)
    if Tset == {"top", "right"}:
        g[0, N - 1] = timing_corner_value
    elif Tset == {"top", "left"}:
        g[0, 0] = timing_corner_value
    elif Tset == {"bottom", "right"}:
        g[N - 1, N - 1] = timing_corner_value
    elif Tset == {"bottom", "left"}:
        g[N - 1, 0] = timing_corner_value

    return g


# ============================================================
# Synthetic output: binary + 2-module quiet zone
# ============================================================

def pad_modules(grid, pad=2):
    N = grid.shape[0]
    out = np.zeros((N + 2 * pad, N + 2 * pad), dtype=np.uint8)  # 0 means white in render
    out[pad:pad + N, pad:pad + N] = grid
    return out


def grid_to_image(grid, scale=12):
    g = (grid > 0).astype(np.uint8)
    img = (1 - g) * 255
    N = img.shape[0]
    out = cv2.resize(img, (N * scale, N * scale), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8)


# ============================================================
# Fallback: border finding via iterative bands + RANSAC on centers
# ============================================================

def normalize_line(L):
    a, b, c = L
    n = float(np.hypot(a, b) + 1e-8)
    return (float(a / n), float(b / n), float(c / n))

def line_vxy_to_abc(L):
    """
    Convert cv2.fitLine format (vx, vy, x0, y0)
    to normalized ax + by + c = 0
    """
    vx, vy, x0, y0 = L
    a = -vy
    b = vx
    c = vy * x0 - vx * y0

    n = np.hypot(a, b)
    if n > 0:
        a /= n
        b /= n
        c /= n

    return (a, b, c)

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


def has_blob_near_point(blobs_xy, p_xy, r):
    if p_xy is None:
        return False
    d = np.linalg.norm(blobs_xy - p_xy[None, :], axis=1)
    return bool(np.any(d <= r))


def choose_L_sides_from_border_counts(counts):
    adj = [("top", "right"), ("right", "bottom"), ("bottom", "left"), ("left", "top")]
    return max(adj, key=lambda p: int(counts.get(p[0], 0)) + int(counts.get(p[1], 0)))


def try_find_border_iterative_locked(
    blobs,
    gray_used,
    N,
    dbg_dir=None,
    stem=None,
    max_iters=8,
    max_depth_cells=3,
    min_L=None,
    min_TP=None
):
    """
    Two-phase fallback WITHOUT restarting:
      Phase 1: find and lock adjacent L sides
      Phase 2: keep searching remaining sides for timing (TP)
              (no depth reset, no last_try reset)

    Key change:
      - Maintain BEST line per side across all band expansions.
      - Locking uses best[s] (not last_try).
      - Debug draws:
          * best candidate per side (magenta)
          * locked L (yellow)
          * locked TP (cyan)

    Blob support counting:
      A blob counts only if the fitted line passes through
      the INNER CORE (≈60%) of the blob.
    """

    h, w = gray_used.shape
    if len(blobs) < 10:
        return None

    blobs_xy = np.stack([blobs[:, 1], blobs[:, 0]], axis=1).astype(np.float32)
    blobs_sigma = blobs[:, 2].astype(np.float32)

    cellN = min(h, w) / float(N)
    band_width = 1.5 * cellN
    band_step  = 0.5 * cellN
    max_depth  = max_depth_cells * cellN

    if min_L is None:
        min_L = int(0.7 * N)
    if min_TP is None:
        min_TP = int(0.4 * N)

    sides = ("top", "right", "bottom", "left")
    adj_pairs = [("top", "left"), ("top", "right"),
                 ("bottom", "left"), ("bottom", "right")]

    depth  = {s: 0.0 for s in sides}
    locked = {s: None for s in sides}
    counts = {s: 0 for s in sides}

    best = {
        s: {
            "vxy": None,
            "abc": None,
            "cnt": 0,
            "depth": None
        }
        for s in sides
    }

    phase = "L"
    L_pair = None

    def fit_line_vxy(pts):
        vx, vy, x0, y0 = cv2.fitLine(
            pts.astype(np.float32),
            cv2.DIST_L2, 0, 0.01, 0.01
        )
        # make scalars (avoids future numpy warnings)
        return float(vx.ravel()[0]), float(vy.ravel()[0]), float(x0.ravel()[0]), float(y0.ravel()[0])

    def count_support(Labc, pts_xy, sig, core_frac=0.6):
        a, b, c = Labc
        denom = np.sqrt(a*a + b*b) + 1e-9
        dist = np.abs(a*pts_xy[:, 0] + b*pts_xy[:, 1] + c) / denom

        radius = np.sqrt(2.0) * sig
        radius = np.clip(radius, 0.25*cellN, 0.9*cellN)
        core = core_frac * radius

        return int(np.sum(dist <= core))

    for it in range(max_iters):

        # ----- UPDATE BEST CANDIDATES (monotonic) -----
        if phase == "L":
            search_sides = sides
        else:
            # keep searching only the non-L sides
            search_sides = [s for s in sides if (L_pair is None or s not in L_pair)]

        for s in search_sides:
            if locked[s] is not None or depth[s] > max_depth:
                continue

            pts_fit, pts_all, sig_all = band_points_with_indices(
                blobs_xy, blobs_sigma,
                s, depth[s], band_width, h, w
            )

            if pts_fit is not None and len(pts_fit) >= 2:
                Lvxy = fit_line_vxy(pts_fit)
                Labc = line_vxy_to_abc(Lvxy)
                cnt = count_support(Labc, pts_all, sig_all)

                # keep ONLY if better than previous best
                if cnt > best[s]["cnt"]:
                    best[s] = {
                        "vxy": Lvxy,
                        "abc": Labc,
                        "cnt": cnt,
                        "depth": depth[s]
                    }

            depth[s] += band_step

        # ----- PHASE 1: LOCK L USING BEST (not last_try) -----
        if phase == "L":
            best_pair = None
            for s0, s1 in adj_pairs:
                if best[s0]["abc"] is None or best[s1]["abc"] is None:
                    continue
                score = best[s0]["cnt"] + best[s1]["cnt"]
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, s0, s1)

            if best_pair is not None:
                _, s0, s1 = best_pair
                if best[s0]["cnt"] >= min_L and best[s1]["cnt"] >= min_L:
                    L_pair = (s0, s1)

                    for s in L_pair:
                        locked[s] = best[s]          # store dict with abc/cnt/depth
                        counts[s] = best[s]["cnt"]

                    phase = "TP"

        # ----- PHASE 2: LOCK TP USING BEST (no reset) -----
        if phase == "TP" and L_pair is not None:
            for s in sides:
                if s in L_pair or locked[s] is not None:
                    continue
                if best[s]["abc"] is not None and best[s]["cnt"] >= min_TP:
                    locked[s] = best[s]
                    counts[s] = best[s]["cnt"]

            

        # ----- DEBUG -----
        if dbg_dir is not None and stem is not None:
            vis = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)

            COL_BEST = (255, 0, 255)   # magenta (best candidate)
            COL_L    = (0, 255, 255)   # yellow (locked L)
            COL_TP   = (255, 255, 0)   # cyan (locked TP)

            # draw best candidate lines (magenta)
            for s in sides:
                if best[s]["abc"] is not None:
                    draw_infinite_line(vis, best[s]["abc"], COL_BEST, 1)

            # draw locked lines (L yellow, TP cyan)
            for s in sides:
                if locked[s] is None:
                    continue
                if L_pair is not None and s in L_pair:
                    draw_infinite_line(vis, locked[s]["abc"], COL_L, 2)
                else:
                    draw_infinite_line(vis, locked[s]["abc"], COL_TP, 2)

            draw_blobs(vis, blobs, (255, 0, 0), 1)

            txt = [
                f"LINEFIT N={N} it={it} phase={phase} L_pair={L_pair}",
                "best_cnts: " + " ".join(f"{s}={best[s]['cnt']}" for s in sides),
                "depths: " + " ".join(f"{s}={depth[s]:.1f}" for s in sides),
                f"min_L={min_L} min_TP={min_TP} bw={band_width:.1f} step={band_step:.1f}"
            ]
            put_debug_text(vis, txt, (10, 18))
            cv2.imwrite(str(dbg_dir / f"{stem}_N{N}_iter{it:02d}.png"), vis)

        if all(locked[s] is not None for s in sides):
            return {
                "locked": locked,
                "counts": counts,
                "L_pair": L_pair,
                "cellN": cellN
            }

    return None


def band_points_with_indices(blobs_xy, blobs_sigma, side, depth, band_width, h, w):
    """
    Returns:
      pts_fit   : (K,2) outermost points for line fitting
      pts_all   : (M,2) all band points for support counting
      sig_all   : (M,) sigmas for pts_all
    """

    if side == "top":
        mask = blobs_xy[:, 1] < depth + band_width
    elif side == "bottom":
        mask = blobs_xy[:, 1] > h - (depth + band_width)
    elif side == "left":
        mask = blobs_xy[:, 0] < depth + band_width
    elif side == "right":
        mask = blobs_xy[:, 0] > w - (depth + band_width)
    else:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))

    pts_all = blobs_xy[mask]
    sig_all = blobs_sigma[mask]

    if len(pts_all) < 2:
        return np.zeros((0, 2)), pts_all, sig_all

    # keep outermost ~50% for fitting
    k = max(3, int(0.5 * len(pts_all)))

    if side == "top":
        idx = np.argsort(pts_all[:, 1])[:k]
    elif side == "bottom":
        idx = np.argsort(pts_all[:, 1])[-k:]
    elif side == "left":
        idx = np.argsort(pts_all[:, 0])[:k]
    else:  # right
        idx = np.argsort(pts_all[:, 0])[-k:]

    pts_fit = pts_all[idx]
    return pts_fit, pts_all, sig_all

def offset_lines_for_crop(lines_abc, N, img_shape, pitch=None):
    """
    Offset borders to define a crop quad.

    FINAL RULE:
    - ALWAYS expand outward by 0.5 * cell_size
    - NEVER offset inward
    - If expansion exceeds image, CLIP quad corners to image bounds
    """

    h, w = img_shape
    sides = ("top", "right", "bottom", "left")

    # normalize lines (ax + by + c = 0 with ||(a,b)|| = 1)
    norm = {}
    for s in sides:
        a, b, c = lines_abc[s]
        d = np.sqrt(a*a + b*b) + 1e-9
        norm[s] = (a/d, b/d, c/d)

    # estimate cell size
    if pitch is not None and np.isfinite(pitch):
        cell_size = float(pitch)
    else:
        # fallback: estimate from opposite borders
        def line_dist(L1, L2):
            return abs(L1[2] - L2[2])

        cell_y = line_dist(norm["top"], norm["bottom"]) / (N - 1)
        cell_x = line_dist(norm["left"], norm["right"]) / (N - 1)
        cell_size = 0.5 * (cell_x + cell_y)

    delta = 0.5 * cell_size  # ALWAYS outward

    center = np.array([w * 0.5, h * 0.5], dtype=np.float32)

    # offset lines outward
    offset = {}
    for s in sides:
        a, b, c = norm[s]
        sign = compute_outward_sign((a, b, c), center)
        offset[s] = (a, b, c + sign * delta)

    # compute quad from offset lines
    quad = quad_from_abc_lines(offset)
    if quad is None:
        return None

    # CLIP quad corners to image bounds (NO inward padding)
    quad_clipped = []
    for (x, y) in quad:
        x = min(max(float(x), 0.0), w - 1.0)
        y = min(max(float(y), 0.0), h - 1.0)
        quad_clipped.append((x, y))

    return np.array(quad_clipped, dtype=np.float32)

def compute_outward_sign(Ln, center_xy):
    """
    For each border line, decide which normal direction points outward (away from center).
    If center is on positive side (ax+by+c > 0), then outward is +normal, else outward is -normal.
    """
    a, b, c = Ln
    val = a * center_xy[0] + b * center_xy[1] + c
    return 1.0 if val > 0 else -1.0


def build_quad_from_lines(locked_lines):
    TL = intersect_lines(locked_lines["top"], locked_lines["left"])
    TR = intersect_lines(locked_lines["top"], locked_lines["right"])
    BR = intersect_lines(locked_lines["bottom"], locked_lines["right"])
    BL = intersect_lines(locked_lines["bottom"], locked_lines["left"])
    if any(p is None for p in (TL, TR, BR, BL)):
        return None
    return np.array([TL, TR, BR, BL], dtype=np.float32)

# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--log_threshold", type=float, default=0.038)
    ap.add_argument("--max_iters", type=int, default=8)
    ap.add_argument("--max_depth_cells", type=float, default=3)

    ap.add_argument("--warp_size", type=int, default=240)
    ap.add_argument("--scale", type=int, default=12)
    ap.add_argument("--quiet", type=int, default=2)

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

        solved = False
        blobs = None
        gray_used = None
        gray_pol = None
        border = None
        locked = None
        pitch0 = None

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

        # -------------------------
        # LASER
        # -------------------------
        if mode == "laser":
            laser_img = process_laser(gray0)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_laser.png"), laser_img)
            if args.debug:
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_laser_debug.png"), laser_img)
            continue

        if mode != "dotpeen":
            continue

        # -------------------------
        # DOTPEEN: polarity fix
        # -------------------------
        gray_pol, hists = quadrant_polarity_fix(gray0)
        if args.debug:
            save_hist_panel(hists, dbg_dir / f"{img_path.stem}_step0_hists.png")
            cv2.imwrite(str(dbg_dir / f"{img_path.stem}_step0_polarity.png"), gray_pol)

        # -------------------------
        # blobs on original crop
        # -------------------------
        blobs, gray_used, _ = detect_dots_log(gray_pol, grid_size_virtual=15, threshold=args.log_threshold)
        if blobs is None or len(blobs) < 10:
            n = 0 if blobs is None else len(blobs)
            print(f"[WARN] too few blobs: {n} -> skip {img_path.name}")
            continue

        if args.debug:
            vis = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
            draw_blobs(vis, blobs, (255, 0, 0), 2)
            cv2.imwrite(str(dbg_dir / f"{img_path.stem}_dp_step2_blobs.png"), vis)

        # -------------------------
        # LINE FITTING (border lines) on original crop
        # (uses N=16, debug stays as-is inside try_find_border_iterative_locked)
        # -------------------------
        if len(blobs) < 20:
            print(f"[FAIL] insufficient blobs for pitch estimation (<20): {img_path.name}")
            border = None
        else:
            pitch0 = estimate_pitch_only(blobs, min_points=20)

            border = try_find_border_iterative_locked(
                blobs=blobs,
                gray_used=gray_used,
                N=15,
                dbg_dir=(dbg_dir if args.debug else None),
                stem=img_path.stem,
                max_iters=args.max_iters,
                max_depth_cells=args.max_depth_cells
            )
            if border is None:
                print(f"[FAIL] line fitting failed: {img_path.name}")
            else:
                locked = border["locked"]
                if any(locked[s] is None for s in ("top", "right", "bottom", "left")):
                    print(f"[FAIL] line fitting incomplete (missing sides): {img_path.name}")
                    border = None

        # -------------------------
        # WARP using fitted borders
        # -------------------------
        warped = None
        if border is not None:
            quad = offset_lines_for_crop(
                lines_abc={s: locked[s]["abc"] for s in ("top", "right", "bottom", "left")},
                N=15,  # keeping your current call as-is
                img_shape=gray_used.shape,
                pitch=pitch0
            )
            if quad is None:
                print(f"[FAIL] quad build failed: {img_path.name}")
            else:
                pad = int(0.25 * min(gray_pol.shape))
                crop, quad_shift, _ = crop_by_quad(gray_pol, quad, pad=pad)
                corners = order_corners(quad_shift)
                warped = warp_to_square_nearest(crop, corners, out_size=args.warp_size)

                if args.debug:
                    cv2.imwrite(str(dbg_dir / f"{img_path.stem}_warp.png"), warped)

        # -------------------------
        # GRID FITTING on warped crop: try N=16 then N=14
        # -------------------------
        if warped is not None:
            blobs_w, gray_w, _ = detect_dots_log(
                warped,
                grid_size_virtual=16,
                threshold=args.log_threshold
            )

            if blobs_w is None or len(blobs_w) < 10:
                n = 0 if blobs_w is None else len(blobs_w)
                print(f"[FAIL] too few blobs on warped ({n}): {img_path.name}")
            else:
                blobs_xy_w = np.stack([blobs_w[:, 1], blobs_w[:, 0]], axis=1).astype(np.float32)
                hw, ww = gray_w.shape

                L_pair = border["L_pair"]

                def side_count(grid, side):
                    if side == "top":
                        return int(grid[0, :].sum())
                    if side == "bottom":
                        return int(grid[-1, :].sum())
                    if side == "left":
                        return int(grid[:, 0].sum())
                    if side == "right":
                        return int(grid[:, -1].sum())
                    return 0

                def fits_L(grid, L_pair, N):
                    c0 = side_count(grid, L_pair[0])
                    c1 = side_count(grid, L_pair[1])
                    ok = (c0 >= (N - 1)) and (c1 >= (N - 1))
                    return ok, c0, c1

                chosen_N = None
                chosen_grid = None

                for N_try in (16, 14):
                    grid_try = fast_grid_fit_whole_image(blobs_xy_w, (hw, ww), N_try)
                    ok, c0, c1 = fits_L(grid_try, L_pair, N_try)

                    if args.debug:
                        vis_g = cv2.cvtColor(gray_w, cv2.COLOR_GRAY2BGR)
                        draw_grid_on_image(vis_g, N_try, (0, 0, 255), 1)
                        draw_blobs(vis_g, blobs_w, (255, 0, 0), 2)
                        lines = [
                            f"GRIDFIT N={N_try} ok={ok}",
                            f"L_pair={L_pair} counts=({L_pair[0]}:{c0}, {L_pair[1]}:{c1}) need>={N_try-1}",
                            f"n_blobs={len(blobs_w)}"
                        ]
                        put_debug_text(vis_g, lines, (10, 18))
                        cv2.imwrite(str(dbg_dir / f"{img_path.stem}_gridfit_N{N_try}.png"), vis_g)

                    if ok:
                        chosen_N = N_try
                        chosen_grid = grid_try
                        break

                if chosen_N is not None:
                    grid_final = enforce_L_and_timing_by_sides(
                        chosen_grid,
                        L_pair=L_pair,
                        timing_corner_value=0
                    )
                    grid_out = pad_modules(grid_final, pad=args.quiet)
                    syn = grid_to_image(grid_out, scale=args.scale)
                    cv2.imwrite(str(out_dir / f"{img_path.stem}_N{chosen_N}_synthetic.png"), syn)
                    solved = True
                else:
                    print(f"[FAIL] grid fitting failed for N=16 and N=14: {img_path.name}")

        # -------------------------
        # LASER FALLBACK
        # -------------------------
        if not solved:
            print(f"[FAIL] dotpeen failed → exporting laser fallback: {img_path.name}")

            laserf_img = process_laser(gray0)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_fallback_laser.png"), laserf_img)

            if args.debug:
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_fallback_laser_debug.png"), laserf_img)

    print("[OK] Done.")


if __name__ == "__main__":
    main()
