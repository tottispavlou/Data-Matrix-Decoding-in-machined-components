import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode
import os

class DataMatrixDetector:
    def __init__(self, debug_dir="debug_outputs"):
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)

    def preprocess_variants(self, gray, name_prefix="img"):
        """Generate different preprocessing variants and save them."""
        variants = []

        # 1. Original grayscale
        variants.append(("gray", gray))

        # 2. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append(("clahe", clahe_img))

        # 3. Top-hat + Black-hat normalization
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        norm = cv2.add(gray, tophat)
        norm = cv2.subtract(norm, blackhat)
        variants.append(("illum_norm", norm))

        # 4. Morphological close (emphasize dots)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_close = cv2.morphologyEx(norm, cv2.MORPH_CLOSE, kernel2)
        variants.append(("morph_close", morph_close))

        # 5. Bilateral filter (denoise but preserve edges)
        denoised = cv2.bilateralFilter(norm, d=9, sigmaColor=75, sigmaSpace=75)
        variants.append(("bilateral", denoised))

        # Save all variants for inspection
        for name, img in variants:
            cv2.imwrite(os.path.join(self.debug_dir, f"{name_prefix}_{name}.jpg"), img)

        return variants

    def attempt_decode(self, img, tag=""):
        """Try decode normal and inverted versions."""
        decoded = decode(img)
        if decoded:
            print(f"Decoded on {tag}")
            return decoded

        # Try inverted
        inverted = cv2.bitwise_not(img)
        decoded = decode(inverted)
        if decoded:
            print(f"Decoded on {tag} (inverted)")
        return decoded

    def detect_and_decode(self, image_path, visualize=True):
        """Main pipeline: apply multiple preprocessings and try decode."""
        original = cv2.imread(image_path)
        if original is None:
            print(f"Could not load image {image_path}")
            return [], None

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        decoded_strings = []
        output_image = original.copy()

        # Try all preprocessing variants
        variants = self.preprocess_variants(gray, name_prefix=os.path.basename(image_path).split('.')[0])
        for tag, var in variants:
            decoded_objs = self.attempt_decode(var, tag)
            if decoded_objs:
                for obj in decoded_objs:
                    data = obj.data.decode("utf-8")
                    rect = obj.rect
                    decoded_strings.append(data)

                    # Draw bounding box
                    cv2.rectangle(
                        output_image,
                        (rect.left, rect.top),
                        (rect.left + rect.width, rect.top + rect.height),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        output_image,
                        data,
                        (rect.left, rect.top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

        # Save visualization
        if decoded_strings and visualize:
            out_path = os.path.join(self.debug_dir, "detected_result.jpg")
            cv2.imwrite(out_path, output_image)
            print(f"Annotated detection saved to {out_path}")

        return decoded_strings, output_image


def main():
    detector = DataMatrixDetector()
    image_path = r"image.jpg"
    decoded_strings, annotated_image = detector.detect_and_decode(image_path)

    if decoded_strings:
        print("Decoded strings:", decoded_strings)
    else:
        print("No Data Matrix detected")

    if annotated_image is not None:
        cv2.imshow("Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
