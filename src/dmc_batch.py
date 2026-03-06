import sys
from pathlib import Path
import csv
import traceback

from run_pipeline import run_pipeline

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PIPELINE_ROOT = Path("pipeline_results_all")
PIPELINE_ROOT.mkdir(exist_ok=True)

def run_batch(
    image_dir: Path,
    output_csv = PIPELINE_ROOT / "batch_results.csv",
):
    image_dir = image_dir.resolve()
    assert image_dir.exists(), "Input directory does not exist"

    time_accumulator = {}
    num_success = 0

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
            decoded, timings = run_pipeline(str(img_path))
            for k, v in timings.items():
                time_accumulator.setdefault(k, []).append(v)

            num_success += 1
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

    print("\nAverage timing over processed images:")
    for k, values in time_accumulator.items():
        avg = sum(values) / len(values)
        print(f"  {k:12s}: {avg:.3f} s")

    if "total" in time_accumulator:
        print(f"\nAverage TOTAL pipeline time: "
            f"{sum(time_accumulator['total']) / len(time_accumulator['total']):.3f} s")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/dmc_batch.py data/raw/final_testing_dataset")
        sys.exit(1)

    run_batch(Path(sys.argv[1]))
