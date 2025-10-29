import argparse
from pathlib import Path
import cv2
import numpy as np

def gaussian_bg_correct(bgr, method="adaptive", ksize=51, sigma=0.0):
    """
    Gaussian background correction with adaptive choice.
    - method='divide': I' = I / (G + eps), rescale 0..255
    - method='subtract': I' = I - G, rescale 0..255
    - method='adaptive': decide based on mean brightness
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    gray_f = gray.astype(np.float32)
    blur_f = blur.astype(np.float32)

    chosen = method
    if method == "adaptive":
        mean_intensity = gray.mean()
        if mean_intensity > 100:   # quite bright → glare
            chosen = "divide"
        else:                      # darker → preserve dots
            chosen = "subtract"

    # if chosen == "divide":
    #     eps = 1e-6
    #     corr = gray_f / (blur_f + eps)
    #     corr -= corr.min()
    #     if corr.max() > 0:
    #         corr *= 255.0 / corr.max()
    if chosen == "divide":
        eps = 1e-6
        ratio = gray_f / (blur_f + eps)
        ratio = np.power(np.clip(ratio, 0, 5), 0.6)
        corr = ratio - ratio.min()
        if corr.max() > 0:
            corr *= 255.0 / corr.max()
    elif chosen == "subtract":
        corr = gray_f - blur_f
        mn, mx = corr.min(), corr.max()
        if mx - mn > 1e-6:
            corr = (corr - mn) * (255.0 / (mx - mn))
        else:
            corr = np.zeros_like(corr)
    else:
        raise ValueError(f"Unknown method: {chosen}")

    out = np.clip(corr, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out_bgr, blur, chosen

def save_debug_panel(orig, blur_gray, processed, out_path, method_used):
    if len(blur_gray.shape) == 2:
        blur_bgr = cv2.cvtColor(blur_gray, cv2.COLOR_GRAY2BGR)
    else:
        blur_bgr = blur_gray
    h = min(orig.shape[0], processed.shape[0])
    orig_r = cv2.resize(orig, (int(orig.shape[1]*h/orig.shape[0]), h))
    blur_r = cv2.resize(blur_bgr, (int(blur_bgr.shape[1]*h/blur_bgr.shape[0]), h))
    proc_r = cv2.resize(processed, (int(processed.shape[1]*h/processed.shape[0]), h))
    panel = np.hstack([orig_r, blur_r, proc_r])
    cv2.putText(panel, f"Method: {method_used}", (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imwrite(str(out_path), panel)

def main():
    ap = argparse.ArgumentParser(description="Batch Gaussian background correction for images.")
    ap.add_argument("--in_dir", required=True, help="Input images root")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--method", choices=["divide","subtract","adaptive"], default="adaptive",
                    help="Background correction method")
    ap.add_argument("--ksize", type=int, default=51, help="Gaussian kernel size (odd)")
    ap.add_argument("--sigma", type=float, default=0.0, help="Gaussian sigma - if 0 OpenCV infers)")
    ap.add_argument("--exts", nargs="+", default=[".jpg",".jpeg",".png"])
    ap.add_argument("--keep_tree", action="store_true",
                    help="Preserve subfolder structure under out_dir")
    ap.add_argument("--debug", action="store_true",
                    help="Save side-by-side panels under out_dir/_debug")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir = out_dir / "_debug"
    if args.debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.lower() for e in args.exts}
    img_paths = [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]

    print(f"[INFO] Found {len(img_paths)} images in {in_dir}")
    count = 0
    for src in img_paths:
        rel = src.relative_to(in_dir)
        dst_folder = (out_dir / rel.parent) if args.keep_tree else out_dir
        dst_folder.mkdir(parents=True, exist_ok=True)
        dst = dst_folder / rel.name

        img = cv2.imread(str(src))
        if img is None:
            print(f"[WARN] Could not read: {src}")
            continue

        processed, blur, chosen = gaussian_bg_correct(img, method=args.method,
                                                     ksize=args.ksize, sigma=args.sigma)
        cv2.imwrite(str(dst), processed)
        count += 1

        if args.debug:
            dbg_path = dbg_dir / (src.stem + f"_panel.jpg")
            save_debug_panel(img, blur, processed, dbg_path, chosen)

    print(f"[OK] Saved {count} preprocessed images to {out_dir}")

if __name__ == "__main__":
    main()
