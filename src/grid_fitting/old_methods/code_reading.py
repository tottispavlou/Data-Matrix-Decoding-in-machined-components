import cv2
import numpy as np
import argparse
from pathlib import Path
from skimage.feature import blob_log

# ------------ DOT-PEENED BLOB DETECTION ------------ #

def detect_dots_log(img, grid_size=14,
                    cell_ratio=0.6,     # dot around 60% of cell size
                    num_sigma=5,        # number of sigma levels
                    threshold=0.03,     # sensitivity
                    overlap=0.5,        # merge overlap
                    debug=False):
    """
    Detect dot-peened DMC blobs using Laplacian of Gaussian (LoG).
    Sigma is adapted to cell size, assuming a tight DMC crop.
    """

    # grayscale
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    img_norm = img_gray.astype(np.float32) / 255.0

    # expected blob
    h, w = img_gray.shape
    cell_size = min(h, w) / grid_size
    dot_diameter = cell_size * cell_ratio
    dot_radius = dot_diameter / 2.0

    # σ based on cell geometry
    sigma_est = dot_radius / np.sqrt(2)
    sigma_min = 0.8 * sigma_est
    sigma_max = 1.2 * sigma_est

    blobs = blob_log(
        img_norm,
        min_sigma=sigma_min,
        max_sigma=sigma_max,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap
    )

    if blobs.size > 0:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    # visualization
    dbg_img = None
    if debug:
        dbg_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for y, x, r in blobs:
            cv2.circle(dbg_img, (int(x), int(y)), int(r), (0, 255, 0), 1)

    return (blobs, dbg_img) if debug else blobs


# ------------ LASER THRESHOLDING ------------ #

def threshold_laser(img_gray):
    """
    Thresholding for laser-etched DMCs.
    Produces dark code on light background.
    """
    blur = cv2.medianBlur(img_gray, 3)
    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 5
    )
    return thr


# ------------ GRID + SIDE ANALYSIS ------------ #

def draw_grid(img, grid_size, color=(0, 0, 255), thickness=1):
    """Draw full NxN grid across the entire image."""
    h, w = img.shape[:2]
    step_x = w / grid_size
    step_y = h / grid_size
    grid_img = img.copy()

    for i in range(1, grid_size):
        x = int(i * step_x)
        cv2.line(grid_img, (x, 0), (x, h), color, thickness)
    for j in range(1, grid_size):
        y = int(j * step_y)
        cv2.line(grid_img, (0, y), (w, y), color, thickness)

    return grid_img


def analyze_grid(img_gray, blobs, grid_size):
    """
    Count how many border cells have at least one blob center inside.
    multiple blobs in same cell only count once.
    """
    if blobs is None or len(blobs) == 0:
        return {"top": 0, "bottom": 0, "left": 0, "right": 0}

    h, w = img_gray.shape
    step_x = w / grid_size
    step_y = h / grid_size

    top_cells, bottom_cells, left_cells, right_cells = set(), set(), set(), set()

    for (y, x, r) in blobs:
        cell_x = int(x // step_x)
        cell_y = int(y // step_y)

        if not (0 <= cell_x < grid_size and 0 <= cell_y < grid_size):
            continue

        if cell_y == 0:
            top_cells.add(cell_x)
        elif cell_y == grid_size - 1:
            bottom_cells.add(cell_x)

        if cell_x == 0:
            left_cells.add(cell_y)
        elif cell_x == grid_size - 1:
            right_cells.add(cell_y)

    return {
        "top": len(top_cells),
        "bottom": len(bottom_cells),
        "left": len(left_cells),
        "right": len(right_cells),
    }


def read_label(label_path: Path):
    """Reads YOLO label, returns (tag, conf) for the most confident line."""
    with open(label_path, "r") as f:
        lines = [l.strip().split() for l in f.readlines() if l.strip()]
    if not lines:
        return None
    pairs = [(int(l[0]), float(l[-1])) for l in lines]
    tag, conf = max(pairs, key=lambda x: x[1])
    return tag, conf


# ------------ MAIN SCRIPT ------------ #

def main():
    ap = argparse.ArgumentParser(description="Check DMC grids and detect sides (dot-peened vs laser).")
    ap.add_argument("--imgs", required=True, help="Path to input rectified crops (.png)")
    ap.add_argument("--labels", required=True, help="Path to YOLO label folder (.txt)")
    ap.add_argument("--out", required=True, help="Output folder for results")
    ap.add_argument("--debug", action="store_true", help="Save debug visualization images")
    args = ap.parse_args()

    img_dir = Path(args.imgs)
    label_dir = Path(args.labels)
    out_dir = Path(args.out)
    dbg_dir = out_dir / "_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    # open log file
    log_path = out_dir / "code_read_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")

    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        log(f"[WARN] no .png images found in {img_dir}")

    for img_path in img_paths:
        name = img_path.stem
        label_path = label_dir / f"{name.replace('_rectified','')}.txt"

        if not label_path.exists():
            log(f"[WARN] missing label for {img_path.name}")
            continue

        label = read_label(label_path)
        if not label:
            log(f"[WARN] empty label file for {img_path.name}")
            continue

        tag, conf = label
        img = cv2.imread(str(img_path))
        if img is None:
            log(f"[WARN] could not read {img_path}")
            continue

        if tag == 0:
            mode = "dotpeen"
        elif tag == 1:
            mode = "laser"
        else:
            log(f"[WARN] unknown tag {tag} for {img_path.name}, skipping")
            continue

        log(f"[INFO] Processing {img_path.name} as {mode} (conf={conf:.3f})")

        if mode == "dotpeen":
            blobs, dbg_blobs = detect_dots_log(img, debug=True)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for gs in [14, 16]:
                counts = analyze_grid(img_gray, blobs, gs)
                half = gs // 2
                L_found = (
                    (counts["left"] >= half and counts["top"] >= half) or
                    (counts["top"] >= half and counts["right"] >= half) or
                    (counts["right"] >= half and counts["bottom"] >= half) or
                    (counts["bottom"] >= half and counts["left"] >= half)
                )
                log(f"  Grid {gs}x{gs}: sides={counts}  L={L_found}")

                if args.debug:
                    dbg_grid = draw_grid(dbg_blobs, gs)
                    cv2.imwrite(str(dbg_dir / f"{name}_grid{gs}.jpg"), dbg_grid)

        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thr = threshold_laser(img_gray)
            log("  Laser: thresholded image saved (no grid analysis).")
            if args.debug:
                cv2.imwrite(str(dbg_dir / f"{name}_threshold.jpg"), thr)

    log("[OK] Done.")
    log_file.close()


if __name__ == "__main__":
    main()
