
import numpy as np
import cv2

def render_rectified(M, cell_px=24, dot_radius_frac=0.35, dot_radius_jitter=0.08,
                     polarity='dark_on_light', overlap_prob=0.1):
    """Render a rectified 'dot-peen-like' image for a bit matrix M.
    - Polarity: 'dark_on_light' (indent looks dark) or 'light_on_dark' (raised/bright)
    - Overlap: small probability to perturb dot shape for neighboring cells
    Returns: img (uint8), centers list [(y,x),...], cell_size_px
    """
    h_cells, w_cells = M.shape
    H = h_cells * cell_px
    W = w_cells * cell_px

    # background (metal-ish gray)
    img = np.ones((H, W), dtype=np.float32) * 0.7

    centers = []
    base_r = cell_px * dot_radius_frac

    for i in range(h_cells):
        for j in range(w_cells):
            cy = int((i + 0.5) * cell_px)
            cx = int((j + 0.5) * cell_px)
            centers.append((cy, cx))
            if M[i, j] == 1:
                r = base_r * (1.0 + np.random.uniform(-dot_radius_jitter, dot_radius_jitter))
                # disk mask
                y, x = np.ogrid[:H, :W]
                mask = (x - cx)**2 + (y - cy)**2 <= r*r

                # radial shading: darker core + slight bright rim for indent
                rr = np.sqrt((x - cx)**2 + (y - cy)**2) / (r + 1e-6)
                rr = np.clip(rr, 0, 1)
                core = 1.0 - rr**2

                if polarity == 'dark_on_light':
                    img[mask] -= 0.45 * core[mask]
                    img = np.clip(img, 0, 1)
                    # add a subtle highlight on one side
                    nx = (x - cx) / (r + 1e-6)
                    ny = (y - cy) / (r + 1e-6)
                    light = np.clip(0.5 + 0.5*(0.6*nx + 0.8*ny), 0, 1)
                    img[mask] += 0.1 * light[mask]
                else:
                    # bright core
                    img[mask] += 0.45 * core[mask]
                    img = np.clip(img, 0, 1)
                    nx = (x - cx) / (r + 1e-6)
                    ny = (y - cy) / (r + 1e-6)
                    light = np.clip(0.5 + 0.5*(0.6*nx + 0.8*ny), 0, 1)
                    img[mask] += 0.15 * light[mask]

                # slight overlap / deformation
                if np.random.rand() < overlap_prob:
                    shift = np.random.randint(-int(r*0.2), int(r*0.2)+1)
                    img = np.roll(img, shift=shift, axis=1)  # simple perturbation

    img = np.clip(img, 0, 1)
    img_u8 = (img * 255).astype(np.uint8)
    # mild smooth
    img_u8 = cv2.GaussianBlur(img_u8, (3,3), 0.8)
    return img_u8, centers, cell_px
