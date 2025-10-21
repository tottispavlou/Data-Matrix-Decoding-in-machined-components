#!/usr/bin/env python3
"""
Apply Super-Resolution to cropped/warped DMC images.

Default:
  input  = ./rectified_crops_warped
  output = ./rectified_crops_sr
  model  = EDSR x3 (.pb) via OpenCV dnn_superres

Usage examples:
  python sr_crops.py
  python sr_crops.py --input rectified_crops_warped --output rectified_crops_sr \
                     --model_path EDSR_x3.pb --model edsr --scale 3 --workers 4
"""

import argparse
import sys
import os
from pathlib import Path
import cv2
from cv2 import dnn_superres
from concurrent.futures import ThreadPoolExecutor, as_completed

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def load_sr(model_path: Path, model_name: str, scale: int):
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(model_name.lower(), int(scale))
    return sr

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_image(path: Path):
    # cv2.imread keeps 8-bit depth; flags=IMREAD_UNCHANGED to preserve channels
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    # Convert 16-bit or float to 8-bit for SR
    if img.dtype != 'uint8':
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # Ensure 3-channel for SR
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def to_original_channels(sr_bgr, orig_path: Path):
    # If original was grayscale, write grayscale; else keep BGR
    orig = cv2.imread(str(orig_path), cv2.IMREAD_UNCHANGED)
    if orig is None:
        return sr_bgr
    if len(orig.shape) == 2 or (len(orig.shape) == 3 and orig.shape[2] == 1):
        gray = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY)
        return gray
    return sr_bgr

def process_one(img_path: Path, out_dir: Path, sr):
    try:
        img = read_image(img_path)
        up = sr.upsample(img)  # Super-resolution
        up = to_original_channels(up, img_path)

        rel = img_path.name
        out_path = out_dir / rel
        # Make sure parent exists (flat dir by default)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), up)
        if not ok:
            raise IOError(f"cv2.imwrite failed for {out_path}")
        return (img_path, out_path, None)
    except Exception as e:
        return (img_path, None, e)

def main():
    ap = argparse.ArgumentParser(description="Apply Super-Resolution to cropped DMC images.")
    ap.add_argument("--input", "-i", type=str, default="rectified_crops_warped",
                    help="Input folder containing cropped/warped images.")
    ap.add_argument("--output", "-o", type=str, default="rectified_crops_sr",
                    help="Output folder for SR images.")
    ap.add_argument("--model_path", type=str, default="models/super_r/EDSR_x3.pb",
                    help="Path to OpenCV SR model (.pb). e.g., EDSR_x2.pb or EDSR_x3.pb")
    ap.add_argument("--model", type=str, default="edsr",
                    choices=["edsr", "espcn", "fsrcnn", "lapsrn"],
                    help="Super-resolution model name.")
    ap.add_argument("--scale", type=int, default=3, choices=[2,3,4,8],
                    help="Upscale factor supported by the model file.")
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of worker threads (0/1 = single-thread).")
    ap.add_argument("--extensions", type=str, default=",".join(sorted(VALID_EXTS)),
                    help="Comma-separated list of file extensions to include.")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    model_path = Path(args.model_path)

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERR] Input folder not found: {in_dir}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"[ERR] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Gather files
    exts = {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
            for e in args.extensions.split(",") if e.strip()}
    files = [p for p in sorted(in_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]

    if not files:
        print(f"[WARN] No images found in {in_dir} with extensions: {sorted(exts)}")
        ensure_dir(out_dir)
        sys.exit(0)

    ensure_dir(out_dir)

    # Load SR once (note: OpenCV's SR object is not guaranteed thread-safe; create per-thread if using workers>0)
    if args.workers and args.workers > 1:
        # Threaded: initialize an SR instance in each thread via a factory
        def worker(img_path):
            sr_local = load_sr(model_path, args.model, args.scale)
            return process_one(img_path, out_dir, sr_local)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(worker, p) for p in files]
            ok, fail = 0, 0
            for f in as_completed(futures):
                img_path, out_path, err = f.result()
                if err is None:
                    ok += 1
                else:
                    fail += 1
                    print(f"[ERR] {img_path.name}: {err}", file=sys.stderr)
    else:
        # Single-threaded: reuse one SR instance
        sr = load_sr(model_path, args.model, args.scale)
        ok, fail = 0, 0
        for p in files:
            img_path, out_path, err = process_one(p, out_dir, sr)
            if err is None:
                ok += 1
            else:
                fail += 1
                print(f"[ERR] {img_path.name}: {err}", file=sys.stderr)

    print(f"[DONE] Processed {ok} images; {fail} failed. Output -> {out_dir.resolve()}")

if __name__ == "__main__":
    main()
