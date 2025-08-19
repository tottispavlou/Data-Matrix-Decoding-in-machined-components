import cv2
from pylibdmtx import pylibdmtx
import matplotlib.pyplot as plt

def decode_with_libdmtx(image_path: str):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try decoding directly
    decoded_objects = pylibdmtx.decode(gray)

    if decoded_objects:
        for obj in decoded_objects:
            print("Decoded:", obj.data.decode("utf-8"))
    else:
        print("No Data Matrix code found.")

    # Show image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    test_image = "data/1G9V68937920005.jpeg" 
    decode_with_libdmtx(test_image)
