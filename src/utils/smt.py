import os
import shutil

set1 = "data/preproc/real_inverted"
set2 = "data/raw/val/images"
output_dir = "data/preproc/val_inv"

def main():
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image filenames from raw/train/images
    raw_image_names = set(os.listdir(set1))

    # Iterate through preproc/real and copy only matches
    for fname in os.listdir(set2):
        if fname in raw_image_names:
            src = os.path.join(set2, fname)
            f1, ext = os.path.splitext(fname)
            f2 = f1 + "_inv" + ext
            dst = os.path.join(output_dir, f2)
            shutil.copy(src, dst)
            print(f"Copied {f2}")

    print("Done.")

if __name__ == "__main__":
    main()
