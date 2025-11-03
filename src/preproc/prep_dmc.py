import argparse
from pathlib import Path
import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import blob_log


# ----------------------- Core Functions ----------------------- #

def _percentile_stretch(img_u8: np.ndarray, lo=1, hi=99) -> np.ndarray:
    p1, p99 = np.percentile(img_u8, (lo, hi))
    out = np.clip((img_u8.astype(np.float32) - p1) * 255.0 / (p99 - p1 + 1e-6), 0, 255)
    return out.astype(np.uint8)


def normalize_dmc(img_gray: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Normalize DMC crop so it ends up as DARK code on LIGHT background with good contrast.
    Uses CLAHE + (center vs border) polarity guess + blob-based polarity confirmation.
    Returns (final_image, debug_dict).
    """
    debug = {}
    img = img_gray.astype(np.uint8)

    # --- Step 1: CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(img)
    debug["clahe"] = eq.copy()

    # --- Step 2: entropy (for optional inspection)
    ent = entropy(eq, disk(5))
    ent_norm = cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    debug["entropy"] = ent_norm

    # --- Step 3: center-vs-border brightness
    h, w = eq.shape
    m1, m2 = int(0.2 * h), int(0.8 * h)
    n1, n2 = int(0.2 * w), int(0.8 * w)
    center_mean = np.mean(eq[m1:m2, n1:n2])
    border_mean = np.mean(np.concatenate([
        eq[:m1, :].ravel(), eq[m2:, :].ravel(), eq[:, :n1].ravel(), eq[:, n2:].ravel()
    ]))
    initial_flip = center_mean > border_mean
    eq = cv2.bitwise_not(eq) if initial_flip else eq.copy()
    eq = _percentile_stretch(eq, 1, 99)
    debug["eq_after_initial"] = eq.copy()
    debug["initial_flip"] = bool(initial_flip)

    # --- Step 4: blob detection for polarity confirmation
    blobs = blob_log(eq / 255.0, min_sigma=1.5, max_sigma=3.5, num_sigma=5, threshold=0.03)
    mask = np.zeros_like(eq, dtype=np.uint8)
    for (y, x, r) in blobs:
        cv2.circle(mask, (int(x), int(y)), int(max(1.0, r * 1.2)), 255, -1)

    mean_in = np.mean(eq[mask == 255]) if np.any(mask) else 0
    mean_out = np.mean(eq[mask == 0]) if np.any(mask == 0) else 0
    blobs_bright = mean_in > mean_out if len(blobs) > 0 else None

    if blobs_bright is True:
        eq = cv2.bitwise_not(eq)
    eq = _percentile_stretch(eq, 2, 98)

    debug.update({
        "final": eq.copy(),
        "blob_mask": mask,
        "blobs": blobs,
        "blobs_bright": blobs_bright,
        "mean_in": mean_in,
        "mean_out": mean_out,
        "center_mean": center_mean,
        "border_mean": border_mean,
    })

    return eq, debug


def save_debug_panel(img_orig, debug: dict, out_path: Path):
    """
    Creates a single side-by-side panel for visual debugging.
    Layout: [original | CLAHE | after_initial | final (+blobs overlay)]
    """
    def to_bgr(im):
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if len(im.shape) == 2 else im

    orig = to_bgr(cv2.normalize(img_orig, None, 0, 255, cv2.NORM_MINMAX))
    clahe = to_bgr(debug.get("clahe", orig))
    after_init = to_bgr(debug.get("eq_after_initial", orig))
    final = to_bgr(debug.get("final", orig))

    # draw blobs overlay on final
    if debug.get("blobs") is not None:
        for (y, x, r) in debug["blobs"]:
            cv2.circle(final, (int(x), int(y)), int(max(1.0, r * 1.2)), (0, 0, 255), 1)

    # resize all panels to same height
    panels = [orig, clahe, after_init, final]
    h_min = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h_min / p.shape[0]), h_min)) for p in panels]
    panel = cv2.hconcat(panels)

    cv2.imwrite(str(out_path), panel)


# ----------------------- CLI Entry Point ----------------------- #

def main():
    ap = argparse.ArgumentParser(description="Normalize DMC crops to dark-on-light with blob polarity correction.")
    ap.add_argument("--in_dir", required=True, help="Input images root")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--exts", nargs="+", default=[".png", ".jpg", ".jpeg"], help="Allowed extensions")
    ap.add_argument("--keep_tree", action="store_true", help="Preserve folder tree under out_dir")
    ap.add_argument("--debug", action="store_true", help="Save side-by-side debug panels to out_dir/_debug")
    ap.add_argument("--skip_dirs", nargs="+", default=[], help="Relative subfolder names under in_dir to skip (match on any level)")
    ap.add_argument("--no_recurse", action="store_true", help="Do not recurse into subfolders")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dbg_dir = out_dir / "_debug"
    if args.debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.lower() for e in args.exts}

    def is_skipped(path: Path) -> bool:
        # check any folder in the relative path against skip list
        rel_parts = path.relative_to(in_dir).parts[:-1]  # folders only
        return any(sd in rel_parts for sd in args.skip_dirs)

    if args.no_recurse:
        # only files directly inside in_dir
        img_paths = [p for p in in_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts and not is_skipped(p)]
    else:
        # recurse but skip any path that passes through a skipped folder
        img_paths = [p for p in in_dir.rglob("*")
                     if p.is_file() and p.suffix.lower() in exts and not is_skipped(p)]

    print(f"[INFO] Found {len(img_paths)} images in {in_dir} (skipping: {args.skip_dirs})")

    count = 0
    for src in img_paths:
        rel = src.relative_to(in_dir)
        dst_folder = (out_dir / rel.parent) if args.keep_tree else out_dir
        dst_folder.mkdir(parents=True, exist_ok=True)
        dst = dst_folder / rel.name

        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read: {src}")
            continue

        processed, debug = normalize_dmc(img)
        cv2.imwrite(str(dst), processed)
        count += 1

        if args.debug:
            dbg_path = dbg_dir / f"{src.stem}_panel.jpg"
            save_debug_panel(img, debug, dbg_path)

    print(f"[OK] Saved {count} preprocessed images to {out_dir}")

    
if __name__ == "__main__":
    main()
