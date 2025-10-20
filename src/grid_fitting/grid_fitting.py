import json
import math
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------- helpers ----------------------

def load_candidates(json_path: Path):
    with open(json_path, "r") as f:
        recs = json.load(f)
    pts = np.array([[r["x"], r["y"]] for r in recs], dtype=np.float32)
    return pts

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def to_int_tuple(xy):
    return (int(round(xy[0])), int(round(xy[1])))

@dataclass
class GridFitResult:
    H: np.ndarray                # homography img->rectified
    rectified: np.ndarray        # square, tight warp (uint8)
    grid_size: int               # 14 or 16
    rotation_k: int              # 0..3 (90° steps) applied after warp
    polarity: int                # +1 bright modules = 1, -1 dark modules = 1
    cell_centers: np.ndarray     # (N,N,2) centers in rectified image space
    cell_values: np.ndarray      # (N,N) binarized 0/1 (after polarity fix)


# ---------------------- coarse orientation via PCA ----------------------

def coarse_u_v_axes(points: np.ndarray):
    """
    PCA to get dominant axes. Returns unit vectors u,v (orthonormal).
    """
    pts = points.astype(np.float32)
    mu = pts.mean(axis=0, keepdims=True)
    X = pts - mu
    cov = (X.T @ X) / max(1, len(pts)-1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # largest eigenvector = first axis
    u = eigvecs[:, np.argmax(eigvals)]
    # enforce a right-handed basis; v = u rotated 90°
    v = np.array([ -u[1], u[0] ], dtype=np.float32)
    # normalize
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return mu.squeeze(), u, v

def rotate_points(points, origin, u, v):
    """
    Express points in the (u,v) coordinate frame: (x,y) -> (s,t)
    """
    rel = points - origin
    s = rel @ u   # projection on u
    t = rel @ v   # projection on v
    return np.stack([s, t], axis=1)

import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class RansacLine:
    p0: np.ndarray     # point on line (2,)
    d: np.ndarray      # unit direction (2,)
    inliers: np.ndarray  # (M,2)

def fit_line_ransac_points(pts, thresh=2.0, max_trials=1000):
    """
    Minimal 2-point RANSAC for a line with L2 distance in pixels.
    Returns RansacLine or None.
    """
    if len(pts) < 2: return None
    pts = np.asarray(pts, np.float32)
    best = None
    rng = np.random.default_rng(123)
    for _ in range(max_trials):
        i,j = rng.choice(len(pts), size=2, replace=False)
        p0, p1 = pts[i], pts[j]
        d = p1 - p0
        n = np.linalg.norm(d)
        if n < 1e-6: continue
        d /= n
        # point-line distance
        v = pts - p0
        # perp component magnitude
        dist = np.abs(v[:,0]*(-d[1]) + v[:,1]*(d[0]))
        inl = pts[dist <= thresh]
        if best is None or len(inl) > len(best.inliers):
            best = RansacLine(p0=p0, d=d.copy(), inliers=inl)
    return best

def angle_between(u,v):
    cu = u / (np.linalg.norm(u)+1e-9)
    cv = v / (np.linalg.norm(v)+1e-9)
    a = np.degrees(np.arccos(np.clip(cu@cv, -1.0, 1.0)))
    return a

def project_on_line(pts, p0, d):
    rel = pts - p0
    t = rel @ d  # 1D coordinate along the line
    return t

def estimate_spacing_from_inliers(inliers, p0, d):
    """Median nearest-neighbour gap along the line (robust to outliers)."""
    t = np.sort(project_on_line(inliers, p0, d))
    if len(t) < 2: return None
    gaps = np.diff(t)
    return float(np.median(gaps)) if len(gaps) else None

def find_L_borders_with_ransac(pts, dist_thresh=2.0, max_trials=1000, ortho_tol=20.0):
    """
    Fit two orthogonal, densely populated lines (the solid 'L' borders).
    Returns (lineA, lineB) or (None,None).
    """
    # first line
    L1 = fit_line_ransac_points(pts, thresh=dist_thresh, max_trials=max_trials)
    if L1 is None or len(L1.inliers) < 10: return None, None
    # subtract inliers of L1, then fit second
    mask = np.ones(len(pts), bool)
    # compute dist to L1 to mask its inliers
    v = pts - L1.p0
    dist = np.abs(v[:,0]*(-L1.d[1]) + v[:,1]*(L1.d[0]))
    mask[dist <= dist_thresh] = False
    remain = pts[mask]
    L2 = fit_line_ransac_points(remain, thresh=dist_thresh, max_trials=max_trials)
    if L2 is None or len(L2.inliers) < 10: return None, None
    # ensure near-orthogonal
    a = angle_between(L1.d, L2.d)
    if abs(90.0 - a) > ortho_tol:
        # swap roles and retry: sometimes second fit lands parallel
        return None, None
    return L1, L2

def line_intersection(p0a, da, p0b, db):
    A = np.stack([da, -db], axis=1)  # [da | -db]
    b = (p0b - p0a)
    # solve p0a + t*da = p0b + s*db
    try:
        ts = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    t = ts[0]
    return p0a + t*da

def build_corners_from_L(Lleft, Lbottom, N, spacing_u, spacing_v):
    """
    Lleft runs along +v (vertical), Lbottom runs along +u (horizontal) or vice-versa.
    spacing_* are per-module spacings along each border in pixels.
    """
    c00 = line_intersection(Lleft.p0, Lleft.d, Lbottom.p0, Lbottom.d)  # L-corner
    if c00 is None: return None
    # choose axis directions so they point into the code area (positive quadrant)
    u = Lbottom.d / (np.linalg.norm(Lbottom.d)+1e-9)
    v = Lleft.d   / (np.linalg.norm(Lleft.d)+1e-9)
    # enforce orthonormal basis
    u = u / (np.linalg.norm(u)+1e-9)
    v = np.array([-u[1], u[0]], np.float32) if abs(u@v) > 0.5 else v
    # corners
    c10 = c00 + u * spacing_u * (N-1)
    c01 = c00 + v * spacing_v * (N-1)
    c11 = c10 + (c01 - c00)
    return np.stack([c00, c10, c11, c01], axis=0).astype(np.float32)

# ---------------------- estimate spacing & cluster rows/cols ----------------------

def cluster_1d(values: np.ndarray, K: int):
    """
    Robust 1D k-means. No initial-labels misuse.
    Returns (centers_sorted, labels_remapped, compactness).
    """
    vals = values.reshape(-1, 1).astype(np.float32)
    n = len(vals)

    if n == 0:
        # nothing to cluster
        return np.array([]), np.array([]), float("inf")

    # If not enough points for K clusters, bail cleanly with a bad score
    if n < K:
        return np.array([]), np.array([]), float("inf")

    # Degenerate case: all values (almost) the same
    if float(vals.max() - vals.min()) < 1e-6:
        # place K tiny-separated centers and assign by nearest
        centers = np.linspace(vals.min(), vals.max() + 1e-3, K, dtype=np.float32).reshape(-1, 1)
        # labels by nearest center
        d = np.abs(vals - centers.T)           # (n, K)
        labels = np.argmin(d, axis=1).astype(np.int32)
        compactness = float(np.sum((vals - centers[labels]) ** 2))
        centers = centers.flatten()
        # sort centers and remap labels to 0..K-1 in order
        order = np.argsort(centers)
        centers = centers[order]
        lut = {int(o): i for i, o in enumerate(order)}
        labels = np.vectorize(lambda x: lut[int(x)])(labels)
        return centers, labels, compactness

    # Normal case: let OpenCV do kmeans with PP seeding
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    compactness, labels, centers = cv2.kmeans(
        vals, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.flatten()

    # sort centers and remap labels so cluster 0 < cluster 1 < ...
    order = np.argsort(centers)
    centers = centers[order]
    lut = {int(o): i for i, o in enumerate(order)}
    labels = np.vectorize(lambda x: lut[int(x)])(labels.flatten())

    return centers, labels, float(compactness)


def try_grid_size(S_coords: np.ndarray, T_coords: np.ndarray, N: int):
    """
    Try fitting an NxN lattice by clustering S and T into N bins.
    Returns a score (lower is better) and clustering.
    If clustering fails (not enough points, degenerate), returns +inf score.
    """
    s_centers, s_lbl, s_cost = cluster_1d(S_coords, N)
    t_centers, t_lbl, t_cost = cluster_1d(T_coords, N)

    # If either side failed (e.g., fewer points than N clusters), mark as bad
    if s_centers.size != N or t_centers.size != N:
        return float("inf"), (s_centers, s_lbl), (t_centers, t_lbl)

    # spacing regularity penalty
    s_sp = np.diff(s_centers)
    t_sp = np.diff(t_centers)
    mean_sp = (np.mean(s_sp) + np.mean(t_sp)) + 1e-6
    irreg = (np.std(s_sp) + np.std(t_sp)) / mean_sp

    score = float(s_cost + t_cost) + 1e3 * float(irreg)
    return score, (s_centers, s_lbl), (t_centers, t_lbl)


# ---------------------- build corners & homography ----------------------

def corners_from_clusters(origin, u, v, s_centers, t_centers):
    """
    Compute four outer corners from min/max cluster centers in (u,v) frame.
    """
    s0, s1 = s_centers[0], s_centers[-1]
    t0, t1 = t_centers[0], t_centers[-1]
    # back to image coords: origin + s*u + t*v
    c00 = origin + s0*u + t0*v
    c10 = origin + s1*u + t0*v
    c11 = origin + s1*u + t1*v
    c01 = origin + s0*u + t1*v
    return np.stack([c00, c10, c11, c01], axis=0).astype(np.float32)  # clockwise

def homography_to_square(corners, out_size=256):
    dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return H


# ---------------------- rotation & polarity from ECC200 finder ----------------------

def split_into_cells(rectified: np.ndarray, N: int):
    h, w = rectified.shape
    # cell centers (N x N x 2)
    ys = np.linspace(0.5*h/N, h - 0.5*h/N, N)
    xs = np.linspace(0.5*w/N, w - 0.5*w/N, N)
    grid_x, grid_y = np.meshgrid(xs, ys)
    centers = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    return centers

def sample_cells_mean(rectified: np.ndarray, centers: np.ndarray, radius_frac=0.35):
    """
    Sample mean intensity in a disk around each center.
    """
    H, W = rectified.shape
    N = centers.shape[0]
    cell_w = W / N
    r = int(max(1, round(radius_frac * cell_w)))
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    mask = (xx*xx + yy*yy) <= r*r
    out = np.zeros((N,N), np.float32)
    for i in range(N):
        for j in range(N):
            cx, cy = centers[i,j]
            cx = int(round(cx)); cy = int(round(cy))
            x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
            y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
            roi = rectified[y0:y1, x0:x1]
            m = mask[(y0 - (cy - r)):(y1 - (cy - r)),
                     (x0 - (cx - r)):(x1 - (cx - r))]
            if roi.size == 0 or m.size == 0:
                out[i,j] = 0.0
            else:
                out[i,j] = float(roi[m].mean())
    return out

def rotate_grid(mat, k):
    return np.rot90(mat, k=k)

def ecc200_finder_score(binary_cells):
    """
    Score how well cells match ECC200 border pattern.
    ECC200: solid 'L' on left and bottom borders; other two borders alternate.
    Returns higher score for better match.
    """
    N = binary_cells.shape[0]
    # expect left col = 1s, bottom row = 1s
    left_ok  = np.mean(binary_cells[:,0])
    bottom_ok= np.mean(binary_cells[-1,:])

    # top row alternating, right col alternating
    alt = np.arange(N) % 2
    top_ok   = 1.0 - np.mean(np.abs(binary_cells[0,:] - alt))
    right_ok = 1.0 - np.mean(np.abs(binary_cells[:, -1] - alt))

    # weight solid L strongly
    score = 2.0*left_ok + 2.0*bottom_ok + top_ok + right_ok
    return float(score)

def choose_rotation_and_polarity(rectified: np.ndarray, N: int):
    # normalize
    img = rectified.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    centers = split_into_cells(img, N)
    means = sample_cells_mean(img, centers)

    best = None
    for polarity in (+1, -1):
        # polarity: if +1, bright=1; if -1, dark=1
        cells_val = means if polarity > 0 else 1.0 - means
        # global threshold (works because cells are already fairly uniform)
        thr = 0.5 * (cells_val.min() + cells_val.max())
        binary = (cells_val >= thr).astype(np.float32)

        for k in range(4):  # 0,90,180,270
            b = rotate_grid(binary, k)
            score = ecc200_finder_score(b)
            if (best is None) or (score > best[0]):
                best = (score, k, polarity, b)

    score, k, pol, bbest = best
    return k, pol, bbest, centers  # centers correspond to unrotated rectified


# ---------------------- master function ----------------------

def fit_and_rectify_from_candidates(
    crop_gray: np.ndarray,
    candidates_xy: np.ndarray,
    possible_sizes=(14,16),
    out_size=256
) -> GridFitResult:
    """
    Main pipeline assembling all steps.
    """
    # 1) coarse axes
    origin, u, v = coarse_u_v_axes(candidates_xy)
    st = rotate_points(candidates_xy, origin, u, v)
    S, T = st[:,0], st[:,1]

    # 2) try sizes
    best = None
    for N in possible_sizes:
        score, (s_c, _), (t_c, _) = try_grid_size(S, T, N)
        if (best is None) or (score < best[0]):
            best = (score, N, s_c, t_c)
    _, N, s_centers, t_centers = best

    # 3) corners & homography
    corners = corners_from_clusters(origin, u, v, s_centers, t_centers)
    H = homography_to_square(corners, out_size)
    rectified = cv2.warpPerspective(crop_gray, H, (out_size, out_size), flags=cv2.INTER_LINEAR)

    # 4) rotation & polarity using ECC200 finder
    k, pol, cells_bin, centers = choose_rotation_and_polarity(rectified, N)
    rect_rot = np.rot90(rectified, k=k)
    # if pol is -1, invert for final “module=1 means bright”
    rect_final = rect_rot.copy()
    if pol < 0:
        rect_final = cv2.bitwise_not(rect_rot)

    # rotate centers as well (for downstream visualization)
    # (coarse but fine for debugging; decoding usually uses rectified image directly)
    cell_centers = centers.copy()
    for _ in range(k):
        # rotate 90°: (i,j) -> (j, N-1-i)
        cell_centers = np.transpose(cell_centers, (1,0,2))[::-1,:,:]

    return GridFitResult(
        H=H,
        rectified=rect_final.astype(np.uint8),
        grid_size=N,
        rotation_k=k,
        polarity=pol,
        cell_centers=cell_centers,
        cell_values=cells_bin.astype(np.uint8),
    )

def rectify_via_L(pts_xy, crop_gray, try_sizes=(14,16), out_size=256):
    """
    pts_xy: (M,2) candidate centers (after detection)
    Returns (rectified, N, H) or (None,None,None) if failed.
    """
    pts = np.asarray(pts_xy, np.float32)
    # 1) find two orthogonal dense lines = the solid 'L'
    L1, L2 = find_L_borders_with_ransac(pts, dist_thresh=2.0, max_trials=1200, ortho_tol=20.0)
    if L1 is None: return None, None, None

    # Decide which is left vs bottom by their image orientation:
    # left ~ "more vertical" (|dx| < |dy|), bottom ~ "more horizontal".
    Lvert, Lhoriz = (L1, L2) if abs(L1.d[0]) < abs(L1.d[1]) else (L2, L1)

    # 2) estimate module spacing along each border from its inliers
    sp_u = estimate_spacing_from_inliers(Lhoriz.inliers, Lhoriz.p0, Lhoriz.d)
    sp_v = estimate_spacing_from_inliers(Lvert.inliers,  Lvert.p0,  Lvert.d)
    if sp_u is None or sp_v is None: return None, None, None

    # 3) try N in {14,16} -> build corners -> warp
    best = None
    for N in try_sizes:
        corners = build_corners_from_L(Lvert, Lhoriz, N, sp_u, sp_v)
        if corners is None: continue
        dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], np.float32)
        H = cv2.getPerspectiveTransform(corners, dst)
        rect = cv2.warpPerspective(crop_gray, H, (out_size, out_size), flags=cv2.INTER_LINEAR)
        # simple quality score: border contrast + straightness (optional: add more)
        score = float(cv2.Sobel(rect, cv2.CV_32F, 1,0,ksize=3).var() + cv2.Sobel(rect, cv2.CV_32F, 0,1,ksize=3).var())
        if best is None or score > best[0]:
            best = (score, N, rect, H)
    if best is None: return None, None, None
    _, Nbest, rectified, Hbest = best
    return rectified, Nbest, Hbest

# ---------------------- CLI wrapper ----------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True, help="Rectified crop image path (grayscale or color).")
    ap.add_argument("--cand_json", type=str, required=True, help="JSON from dot_candidates.py (same stem).")
    ap.add_argument("--out", type=str, default="out/rectified_square.png", help="Output rectified square PNG.")
    ap.add_argument("--size", type=int, default=256, help="Output side length.")
    ap.add_argument("--try_sizes", type=str, default="14,16", help="Comma-separated possible grid sizes.")
    args = ap.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.img)
    pts = load_candidates(Path(args.cand_json))
    if pts.size == 0:
        raise RuntimeError("No candidates in JSON.")

    sizes = tuple(int(x) for x in args.try_sizes.split(","))
    res = fit_and_rectify_from_candidates(img, pts, possible_sizes=sizes, out_size=args.size)

    ensure_dir(Path(args.out))
    cv2.imwrite(args.out, res.rectified)
    print(f"[OK] grid={res.grid_size} rot={res.rotation_k*90}° pol={'bright=1' if res.polarity>0 else 'dark=1'} -> {args.out}")


if __name__ == "__main__":
    main()
