import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.baseline_detection import detect_and_decode

if __name__ == "__main__":
    results = detect_and_decode("data/raw/2J0L17962260003.jpeg")
    print("Decoded results:", results)
