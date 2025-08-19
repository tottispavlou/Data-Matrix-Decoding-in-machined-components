import os
import cv2
import csv
from datetime import datetime
from pylibdmtx import pylibdmtx

def decode_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    results = pylibdmtx.decode(img)
    if results:
        return [res.data.decode("utf-8") for res in results]
    return None

def evaluate_dataset(dataset_path, output_csv="baseline_results.csv"):
    total = 0
    success = 0
    failures = []

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "status", "decoded_values"])

        for fname in os.listdir(dataset_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                total += 1
                path = os.path.join(dataset_path, fname)
                decoded = decode_image(path)

                if decoded:
                    success += 1
                    writer.writerow([fname, "success", "; ".join(decoded)])
                    print(f"[OK] {fname} → {decoded}")
                else:
                    failures.append(fname)
                    writer.writerow([fname, "fail", ""])
                    print(f"[FAIL] {fname} → no Data Matrix found")

    print("\n=== Baseline Evaluation with libdmtx ===")
    print(f"Total images: {total}")
    print(f"Successfully decoded: {success}")
    print(f"Failed to decode: {len(failures)}")
    print(f"Success rate: {(success/total*100) if total else 0:.2f}%")

    if failures:
        print("\nFailed files:")
        for f in failures:
            print("  -", f)

    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    dataset_path = "data/"  # put dataset images here    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"baseline_results_{timestamp}.csv"
    evaluate_dataset(dataset_path)
