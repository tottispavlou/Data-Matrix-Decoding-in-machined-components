import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.baseline_detection import detect_and_decode

if __name__ == "__main__":
    results = detect_and_decode("data/rectified_crops_warped/0-12022091060008_png.rf.5c73f9a178e23142e991c2337f3c2e67_best.png")
    print("Decoded results:", results)
