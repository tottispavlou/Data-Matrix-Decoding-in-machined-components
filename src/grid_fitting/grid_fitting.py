import cv2
import numpy as np
import argparse
from pathlib import Path
from scipy.signal import find_peaks
from skimage.feature import blob_log
import matplotlib.pyplot as plt

# ================================================================
#   1.  LOCAL 2–PEAK HISTOGRAM REMAPPING (glare + polarity fix)
# ================================================================

def remap_patch_two_peak(gray_patch):
    """Remap intensities so that:
       - background peak → dark
       - dot peak → bright
       Uses top 2 histogram peaks in the patch.
    """
    hist, _ = np.histogram(gray_patch.ravel(), bins=256, range=(0, 256))
    peaks, _ = find_peaks(hist, distance=20)

    if len(peaks) < 2:
        # fallback: simple normalize
        return cv2.normalize(gray_patch, None, 0, 255, cv2.NORM_MINMAX)

    # pick strongest 2 peaks
    top2 = peaks[np.argsort(hist[peaks])[-2:]]
    p1, p2 = int(top2[0]), int(top2[1])

    bg = min(p1, p2)
    dot = max(p1, p2)

    dist_bg  = np.abs(gray_patch - bg)
    dist_dot = np.abs(gray_patch - dot)

    # dot-likeness score
    dot_score = dist_bg / (dist_bg + dist_dot + 1e-6)
    dot_score = dot_score ** 3        # sharpen dots

    return (dot_score * 255).astype(np.uint8)


def make_local_dot_map(gray):
    """Splits into 4 quadrants → peak-remap each → stitch."""
    h, w = gray.shape
    hh, ww = h // 2, w // 2

    p1 = remap_patch_two_peak(gray[0:hh,   0:ww])
    p2 = remap_patch_two_peak(gray[0:hh,   ww:w])
    p3 = remap_patch_two_peak(gray[hh:h,   0:ww])
    p4 = remap_patch_two_peak(gray[hh:h,   ww:w])

    top = np.hstack((p1, p2))
    bot = np.hstack((p3, p4))
    return np.vstack((top, bot))


# ================================================================
#   2.  GRID-AWARE BLOB DETECTION (LoG tied to cell size)
# ================================================================

def detect_blobs_grid_aware(mapped_img, grid_size, cell_ratio=0.65):
    """
    Uses LoG blob detection where σ is derived from cell size.
    Ensures blobs are the right diameter relative to grid.
    """
    h, w = mapped_img.shape
    cell = min(h, w) / grid_size

    dot_diam = cell * cell_ratio
    sigma = (dot_diam / 2) / np.sqrt(2)

    min_sigma = sigma * 0.8
    max_sigma = sigma * 1.2

    img_norm = mapped_img.astype(np.float32) / 255.0

    blobs = blob_log(
        img_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=4
    )

    if blobs.size > 0:
        blobs[:, 2] *= np.sqrt(2)

    return blobs


# ================================================================
#   3.  LASER LOCAL THRESHOLDING
# ================================================================

def threshold_laser_local(img_gray):
    """CLAHE + adaptive local inversion thresholding."""
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    cl = clahe.apply(img_gray)

    thr = cv2.adaptiveThreshold(
        cl, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    return thr


# ================================================================
#   4.  GRID SIDE ANALYSIS (your old logic)
# ================================================================

def analyze_grid(img_gray, pts, grid_size):
    """
    Count how many border cells have at least 1 blob.
    pts = list of (x,y)
    """
    h, w = img_gray.shape
    step_x = w / grid_size
    step_y = h / grid_size

    top, bottom, left, right = set(), set(), set(), set()

    for (x, y) in pts:
        cx = int(x // step_x)
        cy = int(y // step_y)
        if not (0 <= cx < grid_size and 0 <= cy < grid_size):
            continue

        if cy == 0:
            top.add(cx)
        elif cy == grid_size - 1:
            bottom.add(cx)

        if cx == 0:
            left.add(cy)
        elif cx == grid_size - 1:
            right.add(cy)

    return {
        "top": len(top),
        "bottom": len(bottom),
        "left": len(left),
        "right": len(right)
    }


# ================================================================
#   5.  READ YOLO LABEL
# ================================================================

def find_label_for_image(img_path, label_dir):
    """
    Takes:   IMGNAME_rectified.png
    Finds:   IMGNAME.txt
    """
    stem = img_path.stem.replace("_rectified", "")
    p = label_dir / f"{stem}.txt"
    return p


def read_label(label_path):
    """Return (tag, confidence) for the top-conf YOLO label line."""
    with open(label_path, "r") as f:
        lines = [l.strip().split() for l in f.readlines() if l.strip()]
    if not lines:
        return None
    pairs = [(int(l[0]), float(l[-1])) for l in lines]
    return max(pairs, key=lambda x: x[1])


# ================================================================
#   6.  DEBUG PANEL
# ================================================================

def make_debug_panel(orig, dotmap, blobs, grid_size):
    """
    Combines:
    [ original | dotmap | blobs ]
    into one horizontal debug panel.
    """
    h, w = orig.shape
    orig_c = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    dot_c  = cv2.cvtColor(dotmap, cv2.COLOR_GRAY2BGR)

    # draw blobs
    blob_img = dot_c.copy()
    for y, x, r in blobs:
        cv2.circle(blob_img, (int(x), int(y)), int(r), (0,0,255), 1)

    panel = np.hstack((orig_c, dot_c, blob_img))
    return panel

def save_local_histograms(img_gray, out_path):
    """
    Saves a panel of 4 local histograms (quadrants) + global histogram.
    """
    h, w = img_gray.shape

    # Quadrants
    q1 = img_gray[0:h//2,     0:w//2]
    q2 = img_gray[0:h//2,     w//2:w]
    q3 = img_gray[h//2:h,     0:w//2]
    q4 = img_gray[h//2:h,     w//2:w]

    quads = [q1, q2, q3, q4]

    plt.figure(figsize=(10, 6))

    # Plot quadrants
    for i, q in enumerate(quads):
        plt.subplot(2, 3, i+1)
        plt.hist(q.ravel(), bins=32, color='gray')
        plt.title(f"Quadrant {i+1}")
        plt.xlim(0, 255)

    # Global histogram
    plt.subplot(2, 3, 5)
    plt.hist(img_gray.ravel(), bins=32, color='black')
    plt.title("GLOBAL")
    plt.xlim(0, 255)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ================================================================
#   7.  MAIN SCRIPT
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True, help="Folder with rectified crops")
    ap.add_argument("--labels", required=True, help="YOLO label folder")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.imgs)
    label_dir = Path(args.labels)
    out_dir = Path(args.out)
    dbg_dir = out_dir / "_debug"

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "code_read_log.txt", "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    img_paths = sorted(img_dir.glob("*.png"))

    for img_path in img_paths:
        # ----------------------
        # load image
        # ----------------------
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            log(f"[WARN] could not read {img_path.name}")
            continue

        # ----------------------
        # find YOLO label
        # ----------------------
        label_path = find_label_for_image(img_path, label_dir)
        if not label_path.exists():
            log(f"[WARN] missing label for {img_path.name}")
            continue

        label = read_label(label_path)
        if label is None:
            log(f"[WARN] empty label for {img_path.name}")
            continue

        tag, conf = label
        mode = "dotpeen" if tag == 0 else "laser" if tag == 1 else "unknown"

        log(f"[INFO] Processing {img_path.name} as {mode} (conf={conf:.3f})")

        # ----------------------
        # DOT-PEENED PIPELINE
        # ----------------------
        if mode == "dotpeen":
            dotmap = make_local_dot_map(img_gray)

            for gs in [14, 16]:
                blobs = detect_blobs_grid_aware(dotmap, gs)
                pts = [(int(b[1]), int(b[0])) for b in blobs]

                counts = analyze_grid(img_gray, pts, gs)
                half = gs // 2
                L_found = (
                    (counts["left"] >= half and counts["top"] >= half) or
                    (counts["top"] >= half and counts["right"] >= half) or
                    (counts["right"] >= half and counts["bottom"] >= half) or
                    (counts["bottom"] >= half and counts["left"] >= half)
                )

                log(f"  Grid {gs}x{gs}: {counts}  L={L_found}")

                if args.debug:
                    panel = make_debug_panel(img_gray, dotmap, blobs, gs)
                    cv2.imwrite(str(dbg_dir / f"{img_path.stem}_grid{gs}.jpg"), panel)
                    histo_path = dbg_dir / f"{img_path.stem}_histo.png"
                    save_local_histograms(img_gray, histo_path)

        # ----------------------
        # LASER PIPELINE
        # ----------------------
        elif mode == "laser":
            thr = threshold_laser_local(img_gray)
            log("  Laser threshold computed.")
            if args.debug:
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_laser.jpg"), thr)

    log("[OK] Done.")
    log_file.close()


if __name__ == "__main__":
    main()
