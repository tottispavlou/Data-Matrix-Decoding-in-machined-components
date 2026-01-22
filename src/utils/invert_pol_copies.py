import os
import cv2

input_folder = "data/raw/val/images"
output_folder = "data/raw/val_inverted"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input directory
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        input_path = os.path.join(input_folder, filename)
        fn, ext = os.path.splitext(filename)
        new_fn = fn + "_inv" + ext
        output_path = os.path.join(output_folder, new_fn)

        # Read image in grayscale mode
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Skipped (not an image): {new_fn}")
            continue

        # Invert image: 255 - pixel
        inverted = 255 - img

        # Save inverted image
        cv2.imwrite(output_path, inverted)

        print(f"Inverted: {filename}")

print("Done!")
