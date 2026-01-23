import sys
from pathlib import Path
import csv
import traceback

from run_pipeline import run_pipeline

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PIPELINE_ROOT = Path("pipeline_results")
PIPELINE_ROOT.mkdir(exist_ok=True)

def run_batch(
    image_dir: Path,
    output_csv = PIPELINE_ROOT / "batch_results.csv",
):
    image_dir = image_dir.resolve()
    assert image_dir.exists(), "Input directory does not exist"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )

    if not images:
        print("No images found")
        return

    results = []

    print(f"[FOUND] {len(images)} images\n")

    for i, img_path in enumerate(images, 1):
        print("=" * 60)
        print(f"[{i}/{len(images)}] Processing: {img_path.name}")

        try:
            decoded = run_pipeline(str(img_path))
            results.append({
                "image": img_path.name,
                "decoded": decoded or "",
                "status": "ok" if decoded else "decode_failed",
            })

        except Exception as e:
            print(f"[ERROR] on {img_path.name}")
            traceback.print_exc()
            results.append({
                "image": img_path.name,
                "decoded": "",
                "status": "error",
            })

    # --------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "decoded", "status"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n Batch processing complete")

# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/dmc_batch.py data/raw/final_testing_dataset")
        sys.exit(1)

    run_batch(Path(sys.argv[1]))
