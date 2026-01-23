import subprocess
import sys
from pathlib import Path
import cv2
import shutil
import time
from pylibdmtx.pylibdmtx import decode
from preproc.preprocess_gaussian import gaussian_bg_correct


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
YOLO_WEIGHTS = Path("models/runs/segment/train_250synth_400/weights/best.pt")

PREPROC_SCRIPT = Path("src/preproc/preprocess_gaussian.py")
RECTIFY_SCRIPT = Path("src/detection/rectify_crops_segm.py")
GRID_SCRIPT = Path("src/grid_fitting/grid_fitting.py")

PIPELINE_ROOT = Path("pipeline_results")
PIPELINE_ROOT.mkdir(exist_ok=True)


# --------------------------------------------------
# LIBDMTX DECODE
# --------------------------------------------------

def decode_dmtx(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    res = decode(img)
    if not res:
        return None

    return res[0].data.decode("ascii", errors="ignore")

# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

def run_pipeline(image_path: str, save_txt=True):
    t0 = time.perf_counter()
    t_stage = time.perf_counter()

    image_path = Path(image_path).resolve()
    assert image_path.exists(), "Image not found"

    print(f"[1] Input image: {image_path}")

    TMP_DIR = PIPELINE_ROOT / f"dmc_pipeline_{image_path.stem}"
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Step 1: Gaussian preprocess
    # --------------------------------------------------
    preproc_img = TMP_DIR / "blurred.png"
    orig = cv2.imread(str(image_path))
    if orig is None:
        raise RuntimeError("Failed to read input image")

    blurred, _, method_used = gaussian_bg_correct(
        orig,
        method="divide",
        ksize=51,
        sigma=0.0
    )

    cv2.imwrite(str(preproc_img), blurred)
    print(f"[2] Gaussian preprocessing done (method={method_used}) -- [TIME]: {time.perf_counter() - t_stage:.3f} s")

    # --------------------------------------------------
    # Step 2: YOLO segmentation
    # --------------------------------------------------
    subprocess.run([
        "yolo",
        "segment",
        "predict",
        f"model={YOLO_WEIGHTS}",
        f"source={preproc_img}",
        "save_txt=True",
        "save_conf=True",
        "save=False",
        f"project={TMP_DIR}",
        "name=yolo"
    ], check=True)

    label_dir = TMP_DIR / "yolo" / "labels"
    labels = list(label_dir.glob("*.txt"))
    if not labels:
        print("[YOLO] No label found")
        return None

    label_path = labels[0]
    print(f"[3] YOLO detection done -- [TIME]: {time.perf_counter() - t_stage:.3f} s")

    # --------------------------------------------------
    # Step 3: Crop & warp
    # --------------------------------------------------
    rectify_img_dir = TMP_DIR / "rectify_imgs"
    rectify_lbl_dir = TMP_DIR / "rectify_labels"
    rectify_out_dir = TMP_DIR / "rectified"

    rectify_img_dir.mkdir(exist_ok=True)
    rectify_lbl_dir.mkdir(exist_ok=True)
    rectify_out_dir.mkdir(exist_ok=True)

    shutil.copy(image_path, rectify_img_dir / image_path.name)
    dst_lbl = rectify_lbl_dir / f"{image_path.stem}.txt"
    shutil.copy(label_path, dst_lbl)
    
    subprocess.run([
        sys.executable,
        "src/detection/rectify_crops_segm.py",
        "--img_dir", rectify_img_dir,
        "--label_dir", rectify_lbl_dir,
        "--out_dir", rectify_out_dir,
        "--reso", "--debug"
    ], check=True)

    print(f"[4] Crop & warp done -- [TIME]: {time.perf_counter() - t_stage:.3f} s")

    rectified_imgs = list(rectify_out_dir.glob("*_rectified.png"))
    if not rectified_imgs:
        raise RuntimeError("No rectified image produced")

    rectified_img = rectified_imgs[0]

    # --------------------------------------------------
    # Step 4: Grid fitting → synthetic DMC
    # --------------------------------------------------
    grid_img_dir = TMP_DIR / "grid_imgs"
    grid_lbl_dir = TMP_DIR / "grid_labels"
    grid_out_dir = TMP_DIR / "grid_out"

    grid_img_dir.mkdir(exist_ok=True)
    grid_lbl_dir.mkdir(exist_ok=True)
    grid_out_dir.mkdir(exist_ok=True)

    shutil.copy(rectified_img, grid_img_dir / rectified_img.name)
    shutil.copy(
        label_path,
        grid_lbl_dir / f"{rectified_img.stem.replace('_rectified','')}.txt"
    )

    synthetic_img = TMP_DIR / "synthetic.png"

    subprocess.run([
        sys.executable,
        "src/grid_fitting/grid_fitting_1_updated.py",
        "--imgs", grid_img_dir,
        "--labels", grid_lbl_dir,
        "--out", grid_out_dir,
        "--debug"
    ], check=True)

    synthetic_imgs = [
        p for p in grid_out_dir.glob("*.png")
        if p.name.endswith("_synthetic.png") or p.name.endswith("_laser.png")
    ]
    if not synthetic_imgs:
        raise RuntimeError("Grid fitting produced no synthetic image")

    synthetic_img = synthetic_imgs[0]

    print(f"[5] Grid fitting done -- [TIME]: {time.perf_counter() - t_stage:.3f} s")

    # --------------------------------------------------
    # Step 5: libdmtx decode
    # --------------------------------------------------
    decoded = decode_dmtx(synthetic_img)

    if decoded:
        print(f"[DECODED]: {decoded}")
        if save_txt:
            out_txt = TMP_DIR / "decoded.txt"
            out_txt.write_text(decoded + "\n")
            print(f"[6] Saved to {out_txt}")
    else:
        print("[FAIL] libdmtx failed")
    
    t1 = time.perf_counter()
    print(f"[TIME] Total pipeline time: {(t1 - t0):.3f} s")

    return decoded

# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/run_pipeline.py data/raw/final_testing_dataset/")
        sys.exit(1)

    run_pipeline(sys.argv[1])
