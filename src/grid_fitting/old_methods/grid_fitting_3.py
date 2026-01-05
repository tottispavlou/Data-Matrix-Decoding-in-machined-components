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
# LoG blobs (dot-peen)
# ============================================================

def detect_dots_log(gray: np.ndarray, grid_size_virtual: int, threshold: float):
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    h, w = gray.shape
    cell = min(h, w) / float(grid_size_virtual)

    dot_radius = 0.5 * cell
    sigma_est = dot_radius / np.sqrt(2.0)
    sigma_min = max(0.6, 0.65 * sigma_est)
    sigma_max = 1.25 * sigma_est

    img_norm = gray.astype(np.float32) / 255.0
    blobs = blob_log(
        img_norm,
        min_sigma=sigma_min,
        max_sigma=sigma_max,
        num_sigma=12,
        threshold=threshold,
        overlap=0.05
    )
    if blobs.size:
        blobs[:, 2] *= np.sqrt(2.0)

    # radius filter to avoid merged / tiny
    r_min = 0.30 * cell
    r_max = 0.55 * cell
    blobs = np.array([b for b in blobs if r_min <= b[2] <= r_max], dtype=np.float32)
    return blobs, gray, cell


# ============================================================
# Geometry helpers
# ============================================================

def line_norm(a, b):
    return float(np.hypot(a, b) + 1e-8)


def normalize_line(L):
    a, b, c = L
    n = line_norm(a, b)
    return (float(a / n), float(b / n), float(c / n))


def dist_point_line(Ln, pts_xy):
    # Ln is normalized (unit normal)
    a, b, c = Ln
    return np.abs(a * pts_xy[:, 0] + b * pts_xy[:, 1] + c)


def signed_point_line(Ln, pts_xy):
    a, b, c = Ln
    return a * pts_xy[:, 0] + b * pts_xy[:, 1] + c


def intersect_lines(L1, L2):
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b1 * (-c2) - b2 * (-c1)) / det
    y = (a2 * (-c1) - a1 * (-c2)) / det
    return np.array([x, y], dtype=np.float32)


def ransac_line_centers(points_xy: np.ndarray, iters=900, dist_thresh=2.5):
    """
    Fit line to blob centers only. Return normalized line and inlier indices.
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) < 2:
        return None, np.array([], dtype=np.int32)

    best_L = None
    best_inl = np.array([], dtype=np.int32)

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
        Ln = normalize_line((a, b, c))

        d = dist_point_line(Ln, pts)
        inl = np.where(d < dist_thresh)[0]
        if len(inl) > len(best_inl):
            best_inl = inl
            best_L = Ln

    return best_L, best_inl


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


# ============================================================
# Border finding with corner checks
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


def has_blob_near_point(blobs_xy, p_xy, r):
    if p_xy is None:
        return False
    d = np.linalg.norm(blobs_xy - p_xy[None, :], axis=1)
    return bool(np.any(d <= r))


def choose_L_sides_from_border_counts(counts):
    adj = [("top", "right"), ("right", "bottom"), ("bottom", "left"), ("left", "top")]
    return max(adj, key=lambda p: int(counts.get(p[0], 0)) + int(counts.get(p[1], 0)))


def corner_of_pair(pair, locked):
    a, b = pair
    if locked.get(a) is None or locked.get(b) is None:
        return None
    return intersect_lines(locked[a], locked[b])


def try_find_border_iterative_locked(
    blobs, gray_used, N,
    dbg_dir=None, stem=None,
    max_iters=18,
    max_depth_cells=1.25,
    corner_r_frac=0.60,
    max_unlocks=6
):
    """
    Per-side search + locking.
    Corner consistency:
      - When adjacent pair (candidate L) locks: their corner MUST have a blob.
        If not, unlock weaker side and continue.
      - When all 4 lock: timing-timing corner MUST be empty. If not, unlock weaker timing side.
    """
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
    lock_min = min_timing  # we lock any side by timing minimum; L vs timing resolved later

    r_corner = corner_r_frac * cellN

    sides = ("top", "right", "bottom", "left")
    locked = {s: None for s in sides}
    locked_counts = {s: 0 for s in sides}
    locked_depth = {s: None for s in sides}
    active = {s: True for s in sides}
    last_try = {s: {"L": None, "cnt": 0, "depth": 0.0} for s in sides}
    unlocks = 0

    def side_status(s):
        if locked[s] is not None:
            return "LOCK"
        if not active[s]:
            return "STOP"
        return "TRY"

    def lock_side(s, L, cnt, depth):
        locked[s] = L
        locked_counts[s] = int(cnt)
        locked_depth[s] = float(depth)
        active[s] = False

    def unlock_side(s):
        nonlocal unlocks
        if locked[s] is None:
            return
        locked[s] = None
        locked_counts[s] = 0
        locked_depth[s] = None
        active[s] = True
        unlocks += 1

    adj_pairs = [("top", "right"), ("right", "bottom"), ("bottom", "left"), ("left", "top")]

    for it in range(max_iters):
        depth = it * band_step

        # stop search if too deep
        for s in sides:
            if locked[s] is not None:
                active[s] = False
            elif active[s] and (depth > max_depth_cells * cellN):
                active[s] = False

        # try lock active sides
        for s in sides:
            if not active[s]:
                continue

            pts = side_band_points(blobs_xy, h, w, s, depth, band_width)
            if len(pts) < lock_min:
                last_try[s] = {"L": None, "cnt": 0, "depth": float(depth)}
                continue

            K = max(min_L, min_timing)
            pts_outer = select_outermost_points(pts, s, K)

            L, inl = ransac_line_centers(pts_outer, iters=900, dist_thresh=dist_thresh)
            cnt = int(len(inl)) if inl is not None else 0
            last_try[s] = {"L": L, "cnt": cnt, "depth": float(depth)}

            if L is not None and cnt >= min_timing:
                lock_side(s, L, cnt, depth)

        # ---------------- Corner checks while iterating ----------------
        if unlocks < max_unlocks:
            # 1) If any adjacent pair is locked, treat best pair as L-candidate and require blob at their corner
            locked_pairs = [p for p in adj_pairs if locked[p[0]] is not None and locked[p[1]] is not None]
            if locked_pairs:
                # choose best candidate by counts
                cand = max(locked_pairs, key=lambda p: locked_counts[p[0]] + locked_counts[p[1]])
                pc = corner_of_pair(cand, locked)
                if pc is not None:
                    if not has_blob_near_point(blobs_xy, pc, r_corner):
                        # unlock weaker side
                        s0, s1 = cand
                        weak = s0 if locked_counts[s0] <= locked_counts[s1] else s1
                        unlock_side(weak)

        # If all 4 locked, enforce timing-timing corner empty
        if all(locked[s] is not None for s in sides):
            L_pair = choose_L_sides_from_border_counts(locked_counts)
            Tset = set(sides) - set(L_pair)
            Tpair = tuple(sorted(Tset, key=lambda x: x))  # just to pick intersection later
            # identify actual adjacent timing pair:
            timing_adj = None
            for p in adj_pairs:
                if p[0] in Tset and p[1] in Tset:
                    timing_adj = p
                    break

            if timing_adj is not None and unlocks < max_unlocks:
                pt = corner_of_pair(timing_adj, locked)
                if pt is not None and has_blob_near_point(blobs_xy, pt, r_corner):
                    # timing-timing corner MUST be empty -> unlock weaker timing side
                    s0, s1 = timing_adj
                    weak = s0 if locked_counts[s0] <= locked_counts[s1] else s1
                    unlock_side(weak)
                else:
                    # success
                    quad = build_quad_from_4lines(locked)
                    if quad is None:
                        return None
                    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
                    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
                    return {
                        "quad": quad,
                        "locked": locked,
                        "counts": locked_counts,
                        "min_L": min_L,
                        "min_timing": min_timing,
                        "cell": cellN
                    }
            else:
                quad = build_quad_from_4lines(locked)
                if quad is None:
                    return None
                quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
                quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
                return {
                    "quad": quad,
                    "locked": locked,
                    "counts": locked_counts,
                    "min_L": min_L,
                    "min_timing": min_timing,
                    "cell": cellN
                }

        # Debug iteration image
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
                f"N={N} it={it} depth={depth:.1f} minL={min_L} minT={min_timing} max_depth={max_depth_cells}cells unlocks={unlocks}",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA
            )
            cv2.imwrite(str(Path(dbg_dir) / f"{stem}_N{N}_iter{it:02d}.png"), vis)

    return None


# ============================================================
# Border forcing
# ============================================================

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
    else:
        Lcorner = (N - 1, N - 1)

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
    # find adjacent timing pair and clear their intersection corner
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
# Warped grid mapping (ideal grid -> image corners)
# ============================================================

def compute_corners_from_locked(locked):
    TL = intersect_lines(locked["top"], locked["left"])
    TR = intersect_lines(locked["top"], locked["right"])
    BR = intersect_lines(locked["bottom"], locked["right"])
    BL = intersect_lines(locked["bottom"], locked["left"])
    if any(p is None for p in (TL, TR, BR, BL)):
        return None
    return TL, TR, BR, BL


def warp_draw_grid(vis, H, N, step=1, color=(0, 0, 255), thickness=1):
    """
    Draw warped grid lines (ideal grid in [0..N-1]) through homography H.
    """
    h, w = vis.shape[:2]

    def proj_pts(P):
        P = np.asarray(P, dtype=np.float32).reshape(-1, 1, 2)
        Q = cv2.perspectiveTransform(P, H).reshape(-1, 2)
        return Q

    # draw horizontal lines
    for i in range(0, N, step):
        src = np.array([[0, i], [N - 1, i]], dtype=np.float32)
        dst = proj_pts(src)
        p0, p1 = dst[0], dst[1]
        cv2.line(vis, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, thickness)

    # draw vertical lines
    for j in range(0, N, step):
        src = np.array([[j, 0], [j, N - 1]], dtype=np.float32)
        dst = proj_pts(src)
        p0, p1 = dst[0], dst[1]
        cv2.line(vis, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, thickness)


def map_blobs_by_inverse_homography(blobs, H, N, tol=0.45):
    """
    Map blob centers into ideal grid coordinates via inv(H),
    then mark black if near integer grid intersection (within tol).
    Returns:
      grid_occ (NxN uint8),
      kept_mask (len(blobs) bool),
      mapped_ij list for debug
    """
    if len(blobs) == 0:
        return np.zeros((N, N), dtype=np.uint8), np.zeros((0,), dtype=bool), []

    Hinv = np.linalg.inv(H)

    pts = np.stack([blobs[:, 1], blobs[:, 0]], axis=1).astype(np.float32)  # [x,y]
    uv = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hinv).reshape(-1, 2)  # [u,v] in ideal grid

    grid = np.zeros((N, N), dtype=np.uint8)
    kept = np.zeros((len(blobs),), dtype=bool)
    mapped = []

    for k, (u, v) in enumerate(uv):
        if not (-1.0 <= u <= (N - 1) + 1.0 and -1.0 <= v <= (N - 1) + 1.0):
            continue

        iu = int(np.round(u))
        iv = int(np.round(v))
        if iu < 0 or iu >= N or iv < 0 or iv >= N:
            continue

        du = abs(u - iu)
        dv = abs(v - iv)
        if du <= tol and dv <= tol:
            grid[iv, iu] = 1  # row=v, col=u
            kept[k] = True
            mapped.append((iv, iu))

    return grid, kept, mapped


# ============================================================
# Synthetic output (binary) + 2-module quiet zone (Option A)
# ============================================================

def pad_modules(grid, pad=2):
    N = grid.shape[0]
    out = np.zeros((N + 2 * pad, N + 2 * pad), dtype=np.uint8)  # white (0 means white in our later render)
    out[pad:pad + N, pad:pad + N] = grid
    return out


def grid_to_image(grid, scale=12):
    g = (grid > 0).astype(np.uint8)
    img = (1 - g) * 255
    N = img.shape[0]
    out = cv2.resize(img, (N * scale, N * scale), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8)


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
    ap.add_argument("--max_iters", type=int, default=8)
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

        # ---- LASER ----
        if mode == "laser":
            thr = process_laser(gray0)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_laser.png"), thr)
            if args.debug:
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_laser_debug.png"), thr)
            continue

        if mode != "dotpeen":
            continue

        # ---- Polarity fix ----
        gray_pol, hists = quadrant_polarity_fix(gray0)
        if args.debug:
            save_hist_panel(hists, dbg_dir / f"{img_path.stem}_step0_hists.png")
            cv2.imwrite(str(dbg_dir / f"{img_path.stem}_step0_polarity.png"), gray_pol)

        # ---- Dots ----
        blobs, gray_used, _ = detect_dots_log(gray_pol, grid_size_virtual=16, threshold=args.log_threshold)
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

            locked = border["locked"]
            counts = border["counts"]

            # determine L pair (for forcing + also semantics)
            L_pair = choose_L_sides_from_border_counts(counts)

            # compute corners in consistent order TL,TR,BR,BL
            corners = compute_corners_from_locked(locked)
            if corners is None:
                continue
            TL, TR, BR, BL = corners

            # homography from ideal grid coords -> image coords
            src = np.array([[0, 0], [N - 1, 0], [N - 1, N - 1], [0, N - 1]], dtype=np.float32)
            dst = np.array([TL, TR, BR, BL], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)

            # map blobs via inverse homography to integer grid intersections
            grid_occ, kept_mask, _ = map_blobs_by_inverse_homography(blobs, H, N, tol=0.45)

            # debug overlays: warped grid + blobs
            if args.debug:
                vis_all = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
                warp_draw_grid(vis_all, H, N, step=1, color=(0, 0, 255), thickness=1)
                draw_blobs(vis_all, blobs, (255, 0, 0), 2)
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_N{N}_final_grid_all.png"), vis_all)

                vis_kept = cv2.cvtColor(gray_used, cv2.COLOR_GRAY2BGR)
                warp_draw_grid(vis_kept, H, N, step=1, color=(0, 0, 255), thickness=1)
                blobs_kept = blobs[kept_mask] if len(kept_mask) == len(blobs) else blobs
                draw_blobs(vis_kept, blobs_kept, (0, 255, 0), 2)
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_N{N}_final_grid_kept.png"), vis_kept)

            # force borders (your rules)
            grid_final = enforce_L_and_timing_by_sides(grid_occ, L_pair)

            # quiet zone (2 modules)
            grid_out = pad_modules(grid_final, pad=args.quiet)
            syn = grid_to_image(grid_out, scale=args.scale)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_N{N}_synthetic.png"), syn)

            solved = True
            break  # IMPORTANT: if N=16 succeeds, do not try N=14

        if not solved:
            print(f"[FAIL] could not solve {img_path.name}")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
