"""
Dot candidate detection on rectified crops with AUTO routing from YOLO labels.

New:
  * --mode auto (default): reads the most-confident class from --label_dir
      class 0 -> dot-peened (DoG pipeline, tuned defaults)
      class 1 -> laser      (segmentation + connected-components centroids)
  * --label_dir: where YOLO OBB txt files live.
  * Handles image<->label name mapping where images may have a trailing "_best"
    (e.g., foo_best.png -> foo.txt)

Outputs (under --out_dir):
  - overlays/*.png  : debug visualization (circles on detected dots)
  - json/*.json     : [{"x":float, "y":float, "r":float, "score":float}, ...]
  - csv/*.csv       : x,y,r,score rows per image (same basename)

Typical usage:
  python dot_candidates.py \
      --in_dir data/rectified_crops_warped \
      --label_dir runs/obb/predict3/labels \
      --out_dir data/dot_candidates_auto \
      --mode auto
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

def normalize_dark_on_light(img_gray: np.ndarray) -> np.ndarray:
    """
    Return an image where code modules are DARK and background is LIGHT.
    Uses Otsu on a lightly blurred image; inverts if the 'foreground' is bright.
    """
    g = cv2.GaussianBlur(img_gray, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If the foreground (0s) is smaller than background, bw==0 are the modules.
    # We want modules to be dark: bw==0 should correspond to 'dots/blocks'.
    # If Otsu made modules white, invert.
    # Heuristic: compare mean intensity inside dark vs light regions.
    m_dark  = float(g[bw == 0].mean()) if np.any(bw == 0) else 0.0
    m_light = float(g[bw == 255].mean()) if np.any(bw == 255) else 255.0
    out = img_gray.copy()
    # If dark region is actually brighter (laser glare or peened highlights), invert.
    if m_dark > m_light:
        out = cv2.bitwise_not(out)
    return out


# -------------------------- utilities --------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def ensure_dirs(base: Path):
    (base / "overlays").mkdir(parents=True, exist_ok=True)
    (base / "json").mkdir(parents=True, exist_ok=True)
    (base / "csv").mkdir(parents=True, exist_ok=True)


def normalize_01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx <= mn:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)


def auto_polarity(resp: np.ndarray) -> int:
    """
    Decide whether dots are bright (+1) or dark (-1) in the response map.
    Heuristic: compare top quantile mean vs bottom quantile mean.
    Returns +1 for 'use positive peaks', -1 for 'use negative peaks'.
    """
    r = resp.flatten().astype(np.float32)
    if r.size < 16:
        return +1
    q_hi = np.quantile(r, 0.90)
    q_lo = np.quantile(r, 0.10)
    hi = r[r >= q_hi].mean()
    lo = r[r <= q_lo].mean()
    return +1 if hi - lo >= 0 else -1


def non_max_suppression(score: np.ndarray, min_dist: int) -> np.ndarray:
    """
    Fast NMS with grayscale dilation. Returns a mask of local-maximum pixels.
    """
    k = 2 * int(min_dist) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    maxima = cv2.dilate(score, kernel)
    return (score >= maxima).astype(np.uint8)


def pick_top_points(score: np.ndarray, mask_nms: np.ndarray, thr_abs: float,
                    thr_rel: float, max_kpts: int) -> list[tuple[int, int, float]]:
    """
    From a (non-negative) score map + NMS mask, threshold and keep top-k.
    Returns list of (x, y, s) with s in [0..1].
    """
    s_norm = normalize_01(score)
    t = max(thr_abs, float(s_norm.max()) * thr_rel)
    keep = (s_norm >= t).astype(np.uint8) & mask_nms
    ys, xs = np.where(keep)
    vals = s_norm[ys, xs]
    if xs.size == 0:
        return []
    order = np.argsort(-vals)  # descending by score
    pts = []
    for idx in order[:max_kpts]:
        pts.append((int(xs[idx]), int(ys[idx]), float(vals[idx])))
    return pts


def estimate_radius_from_scale(sigma_small: float, sigma_large: float | None) -> float:
    """
    Rough mapping of Gaussian sigma to blob radius. Works well for circular-ish dots.
    """
    if sigma_large is None:
        s = sigma_small
    else:
        s = 0.5 * (sigma_small + sigma_large)
    return 1.5 * s  # empirical factor


# -------------------------- DoG / LoG responses --------------------------

def dog_response(img: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    g1 = cv2.GaussianBlur(img, (0, 0), sigma_small)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma_large)
    resp = g1.astype(np.float32) - g2.astype(np.float32)
    return resp


def log_response(img: np.ndarray, sigma: float) -> np.ndarray:
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return lap


# -------------------------- Label helpers --------------------------

def map_image_to_label_path(img_path: Path, label_dir: Path) -> Path | None:
    """
    Map an image filename to its label file.
    Primary attempt: <stem>.txt
    Secondary attempt: if stem endswith "_best" -> strip it and try again.
    """
    exact = label_dir / f"{img_path.stem}.txt"
    if exact.exists():
        return exact
    # handle trailing "_best"
    if img_path.stem.endswith("_best"):
        alt = label_dir / (img_path.stem[:-5] + ".txt")
        if alt.exists():
            return alt
    return None


def read_top_class_from_label(label_path: Path) -> int | None:
    """
    YOLO OBB label format assumed:
      <class> <confidence> x1 y1 x2 y2 x3 y3 x4 y4
    Choose the class with the highest confidence in the file.
    """
    best_conf = -1.0
    best_cls = None
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                cls = int(float(parts[0]))
                conf = float(parts[1])
                if conf > best_conf:
                    best_conf = conf
                    best_cls = cls
    except Exception:
        return None
    return best_cls


# -------------------------- LASER pipeline (segmentation + centroids) --------------------------

def laser_centroid_candidates(
    img_gray: np.ndarray,
    otsu_invert: bool = False,
    open_ksize: int = 2,
    min_area: int = 6,
    max_area: int | None = None,
) -> list[tuple[int, int, float, float]]:
    """
    Segment block-like laser cells and return centroids.
    Returns list of (x, y, r, score) where:
      - r is an equivalent radius from area (sqrt(A/pi))
      - score is normalized area in [0..1]
    """
    H, W = img_gray.shape[:2]

    base = img_gray.copy()
    # mild smoothing to reduce pixel noise
    base = cv2.GaussianBlur(base, (3, 3), 0)

    # Otsu threshold (optionally inverted depending on polarity)
    if otsu_invert:
        _, bw = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # small opening to remove specks / connect cells
    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    # connected components
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return []

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    if max_area is None:
        # heuristic: avoid selecting huge background chunks
        max_area = int(0.08 * H * W)

    # normalize area for scoring
    a_min = max(min_area, 1)
    a_max = max(max_area, a_min + 1)
    pts: list[tuple[int, int, float, float]] = []

    for idx in range(1, num):
        a = int(stats[idx, cv2.CC_STAT_AREA])
        if a < min_area or a > max_area:
            continue
        cx, cy = centroids[idx]
        r_eq = float(np.sqrt(a / np.pi))
        # area-based score 0..1
        score = (a - a_min) / (a_max - a_min)
        score = float(np.clip(score, 0.0, 1.0))
        pts.append((int(round(cx)), int(round(cy)), r_eq, score))

    return pts


# -------------------------- main per-image processing --------------------------

def write_outputs(img: np.ndarray, out_dir: Path, img_path: Path,
                  pts_r: list[tuple[int, int, float, float]], info_text: str):
    """
    pts_r: list of (x, y, r, score)
    """
    # JSON
    json_path = out_dir / "json" / (img_path.stem + ".json")
    records = [{"x": x, "y": y, "r": float(r), "score": float(s)} for (x, y, r, s) in pts_r]
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    # CSV
    csv_path = out_dir / "csv" / (img_path.stem + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "r", "score"])
        for x, y, r, s in pts_r:
            w.writerow([x, y, f"{r:.3f}", f"{s:.4f}"])

    # Overlay
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, r, _ in pts_r:
        cv2.circle(overlay, (x, y), max(1, int(round(r))), (0, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(overlay, info_text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 50), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "overlays" / (img_path.stem + ".png")), overlay)


def process_image_auto_or_forced(
    img_path: Path,
    out_dir: Path,
    mode: str,
    label_dir: Path | None,

    # DoG/LoG params
    method: str,
    sigma_small: float,
    sigma_large: float,
    log_sigma: float,
    min_dist: int,
    thr_abs: float,
    thr_rel: float,
    max_kpts: int,
    clahe: bool,
    equalize: bool,

    # auto presets
    dog_preset: dict,
    laser_preset: dict,
) -> int:
    """
    Process one image according to mode ('auto'|'dog'|'log'|'laser').
    Returns number of candidates written.
    """
    # --- read / grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0

    # base normalization
    base = img.copy()
    if equalize:
        base = cv2.equalizeHist(base)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        base = c.apply(base)

    chosen_mode = mode
    # AUTO routing via label file
    if mode == "auto" and label_dir is not None:
        lbl_path = map_image_to_label_path(img_path, label_dir)
        if lbl_path is not None and lbl_path.exists():
            cls = read_top_class_from_label(lbl_path)
            if cls == 0:
                chosen_mode = "dog"
                # override with tuned DoG defaults
                method = "dog"
                sigma_small = dog_preset.get("sigma_small", sigma_small)
                sigma_large = dog_preset.get("sigma_large", sigma_large)
                min_dist = dog_preset.get("min_dist", min_dist)
                thr_rel = dog_preset.get("thr_rel", thr_rel)
            elif cls == 1:
                chosen_mode = "laser"
                # presets will be used inside laser branch
            else:
                chosen_mode = method  # fallback
        else:
            chosen_mode = method  # fallback (no label)

    # ---------------- DoG / LoG path (dot-peened or manual)
    if chosen_mode in ("dog", "log"):
        # slight specular noise suppression for dot-peen
        base_denoised = cv2.medianBlur(base, 3)

        if chosen_mode == "log" or method == "log":
            base = normalize_dark_on_light(base)
            bg = cv2.blur(base, (1, 31))
            base = cv2.subtract(base, bg)
            base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
            resp = log_response(base_denoised, log_sigma)
            r_est = estimate_radius_from_scale(log_sigma, None)
            header = f"LOG  "
        else:
            base = normalize_dark_on_light(base)
            bg = cv2.blur(base, (1, 31))
            base = cv2.subtract(base, bg)
            base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
            resp = dog_response(base_denoised, sigma_small, sigma_large)
            r_est = estimate_radius_from_scale(sigma_small, sigma_large)
            header = f"DOG  "

        # choose polarity to make dots positive
        pol = auto_polarity(resp)
        if pol < 0:
            resp = -resp

        # clamp/normalize for NMS
        resp_pos = np.clip(resp, 0, None)
        resp_pos = normalize_01(resp_pos)

        # NMS
        nms_mask = non_max_suppression((resp_pos * 255).astype(np.uint8), min_dist=min_dist)

        # threshold & pick
        pts = pick_top_points(resp_pos, nms_mask, thr_abs=thr_abs, thr_rel=thr_rel, max_kpts=max_kpts)

        pts_r = [(x, y, float(max(1.0, r_est)), s) for (x, y, s) in pts]
        info = f"{header}n={len(pts_r)}  pol={'bright' if pol>0 else 'dark'}"
        write_outputs(img, out_dir, img_path, pts_r, info)
        return len(pts_r)

    # ---------------- LASER path (segmentation + centroids)
    elif chosen_mode == "laser":
        # optional polarity guess for Otsu inversion:
        # if brighter cells dominate, invert=False; otherwise try invert=True.
        # Simple heuristic via mean:
        otsu_invert = False
        if float(base.mean()) < 128.0:
            # darker background -> cells might be bright
            otsu_invert = False
        else:
            otsu_invert = True

        base = normalize_dark_on_light(base)
        bg = cv2.blur(base, (1, 31))
        base = cv2.subtract(base, bg)
        base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
        pts_r = laser_centroid_candidates(
            img_gray=base,
            otsu_invert=laser_preset.get("otsu_invert", otsu_invert),
            open_ksize=laser_preset.get("open_ksize", 2),
            min_area=laser_preset.get("min_area", 6),
            max_area=laser_preset.get("max_area", None),
        )

        info = f"LASER n={len(pts_r)}"
        write_outputs(img, out_dir, img_path, pts_r, info)
        return len(pts_r)

    else:
        # unknown mode; do nothing
        return 0


# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/rectified_crops_warped",
                    help="Folder with rectified crops.")
    ap.add_argument("--out_dir", type=str, default="data/dot_candidates",
                    help="Output folder.")
    ap.add_argument("--label_dir", type=str, default="runs/obb/predict3/labels",
                    help="Folder with YOLO OBB label txt files.")
    ap.add_argument("--mode", type=str, default="auto", choices=["auto", "dog", "log", "laser"],
                    help="Routing mode: auto selects pipeline from label class; or force a mode.")

    # DoG/LoG params (used directly in dog/log or as fallback defaults)
    ap.add_argument("--method", type=str, default="dog", choices=["dog", "log"],
                    help="Detector when mode is not 'laser'. Ignored in auto if class decides.")
    ap.add_argument("--sigma_small", type=float, default=1.2, help="DoG small sigma (dot-peened).")
    ap.add_argument("--sigma_large", type=float, default=2.8, help="DoG large sigma (dot-peened).")
    ap.add_argument("--log_sigma", type=float, default=1.4, help="LoG sigma (rarely ideal for laser).")
    ap.add_argument("--min_dist", type=int, default=4, help="NMS min distance (px).")
    ap.add_argument("--thr_abs", type=float, default=0.0, help="Absolute threshold on normalized response [0..1].")
    ap.add_argument("--thr_rel", type=float, default=0.35, help="Relative threshold as fraction of max response.")
    ap.add_argument("--max_kpts", type=int, default=2048, help="Max candidates per image.")

    # Light preprocessing
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE before response.")
    ap.add_argument("--no_equalize", action="store_true", help="Disable global histogram equalization.")

    # Laser presets (only used in 'laser' mode or auto->laser)
    ap.add_argument("--laser_open", type=int, default=2, help="Morph open kernel size for laser mode.")
    ap.add_argument("--laser_min_area", type=int, default=6, help="Min area for connected components (laser).")
    ap.add_argument("--laser_max_area", type=int, default=0, help="Max area (0 = auto heuristic).")
    ap.add_argument("--laser_invert", action="store_true", help="Force Otsu inversion in laser mode.")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    label_dir = Path(args.label_dir) if args.label_dir else None

    imgs = list_images(in_dir)
    if not imgs:
        print(f"[ERR] No images found in {in_dir}")
        return

    eq = not args.no_equalize

    # presets used by AUTO routing
    dog_preset = dict(
        sigma_small=args.sigma_small,
        sigma_large=args.sigma_large,
        min_dist=args.min_dist,
        thr_rel=args.thr_rel,
    )
    laser_preset = dict(
        open_ksize=args.laser_open,
        min_area=args.laser_min_area,
        max_area=(None if args.laser_max_area == 0 else int(args.laser_max_area)),
        otsu_invert=bool(args.laser_invert),
    )

    total = 0
    for p in imgs:
        n = process_image_auto_or_forced(
            img_path=p,
            out_dir=out_dir,
            mode=args.mode,
            label_dir=label_dir,

            method=args.method,
            sigma_small=args.sigma_small,
            sigma_large=args.sigma_large,
            log_sigma=args.log_sigma,
            min_dist=args.min_dist,
            thr_abs=args.thr_abs,
            thr_rel=args.thr_rel,
            max_kpts=args.max_kpts,
            clahe=args.clahe,
            equalize=eq,

            dog_preset=dog_preset,
            laser_preset=laser_preset,
        )
        total += n

    print(f"[OK] Processed {len(imgs)} images. Avg candidates/image: {total/len(imgs):.1f}")


if __name__ == "__main__":
    main()
