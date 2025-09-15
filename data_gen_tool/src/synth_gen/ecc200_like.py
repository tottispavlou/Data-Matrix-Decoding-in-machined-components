
import numpy as np
import random

def make_faux_datamatrix(size: int) -> np.ndarray:
    """Create an ECC200-like bit matrix (size x size).
    - Solid 'L' finder on left and bottom edges
    - Alternating timing on top and right edges
    - Random interior bits
    This is NOT a real ECC200 encoder, but preserves structural cues needed for grid-based methods.
    """
    assert size >= 10 and size % 2 == 0, "Use typical ECC200 sizes like 14 or 16"
    M = np.zeros((size, size), dtype=np.uint8)

    # Finder: left column and bottom row solid (1)
    M[:, 0] = 1
    M[-1, :] = 1

    # Timing: top row and right column alternating 1/0 starting with 1 at (0, size-1) and (0, 0)
    for j in range(size):
        M[0, j] = 1 if j % 2 == 0 else 0
    for i in range(size):
        M[i, -1] = 1 if i % 2 == 0 else 0

    # Interior random
    for i in range(1, size-1):
        for j in range(1, size-1):
            if i == size-1 or j == 0 or i == 0 or j == size-1:
                continue
            M[i, j] = random.randint(0, 1)
    return M
