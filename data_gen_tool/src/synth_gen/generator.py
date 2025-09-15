
import os, math, random, json
from pathlib import Path
import numpy as np
import cv2

from .config import SynthConfig
from .ecc200_like import make_faux_datamatrix
from .metal import compose_background, add_glare_multi, add_scratches
from .realtex import load_texture_paths
from .dotpeen import render_rectified
from .geom import warp_perspective, yolo_box_for_centered
from .labels import write_yolo_label, write_grid_json

class SynthGenerator:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.out_dir = Path(cfg.out_dir)
        (self.out_dir/"images"/"rect").mkdir(parents=True, exist_ok=True)
        (self.out_dir/"images"/"scene").mkdir(parents=True, exist_ok=True)
        (self.out_dir/"labels"/"yolo").mkdir(parents=True, exist_ok=True)
        (self.out_dir/"labels"/"grid").mkdir(parents=True, exist_ok=True)

    def _maybe(self, p): return np.random.rand() < p

    
def _render_scene(self, rect_img):
    # compose on metal background larger than rect_img
    h, w = rect_img.shape
    scale = self.cfg.scene_scale
    H, W = int(h*scale), int(w*scale)

    # set up texture cache once
    if not hasattr(self, "_tex_cache"):
        self._tex_cache = load_texture_paths(self.cfg.metal_textures_dir)

    bg = compose_background(H, W, self.cfg, self._tex_cache)

    # paste rect_img roughly centered with slight blending
    y0 = (H - h)//2
    x0 = (W - w)//2
    scene = bg.copy()
    scene[y0:y0+h, x0:x0+w] = cv2.addWeighted(scene[y0:y0+h, x0:x0+w], 0.3, rect_img, 0.7, 0)

    # optional multi-stripe glare
    if self._maybe(self.cfg.glare_prob):
        layers = np.random.randint(self.cfg.glare_layers[0], self.cfg.glare_layers[1]+1)
        scene = add_glare_multi(scene, layers=layers,
                                strength_range=self.cfg.glare_strength_range,
                                width_range=self.cfg.glare_width_range)

    # optional scratches
    if self._maybe(self.cfg.scratch_prob):
        scene = add_scratches(scene,
                              count=np.random.randint(self.cfg.scratch_count_range[0], self.cfg.scratch_count_range[1]+1),
                              length=np.random.randint(self.cfg.scratch_length_range[0], self.cfg.scratch_length_range[1]+1),
                              thickness=np.random.randint(self.cfg.scratch_thickness_range[0], self.cfg.scratch_thickness_range[1]+1))

    # mild sensor noise
    noise = np.random.normal(0, 3.0, size=scene.shape).astype(np.float32)
    scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if self._maybe(self.cfg.perspective_prob):
        scene, Hmat = warp_perspective(scene, strength=np.random.uniform(0.05, 0.18))

    return scene


    def generate(self):
        index = []
        for idx in range(self.cfg.n_samples):
            size = random.choice(self.cfg.grid_sizes)
            M = make_faux_datamatrix(size)

            # Polarity choice
            polarity = 'dark_on_light' if random.random() > self.cfg.polarity_prob else 'light_on_dark'

            rect_img, centers, cell_px = render_rectified(
                M, cell_px=self.cfg.cell_px,
                dot_radius_frac=self.cfg.dot_radius_frac,
                dot_radius_jitter=self.cfg.dot_radius_jitter,
                polarity=polarity,
                overlap_prob=self.cfg.overlap_prob
            )

            # Optional illumination flattening OFF here; we want raw variety
            # Save rectified crop
            rect_path = self.out_dir/"images"/"rect"/f"synth_{idx:05d}.png"
            cv2.imwrite(str(rect_path), rect_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            # Make scene and yolo label
            scene = self._render_scene(rect_img)
            scene_path = self.out_dir/"images"/"scene"/f"synth_{idx:05d}.jpg"
            cv2.imwrite(str(scene_path), scene, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])

            # YOLO box for full scene (we know the rect is centered and occupies 1/scale)
            h, w = scene.shape
            pad = 0
            cx, cy, ww, hh = yolo_box_for_centered(scene, pad=0)
            yolo_path = self.out_dir/"labels"/"yolo"/f"synth_{idx:05d}.txt"
            # single class 0: "dotpeen"
            write_yolo_label(yolo_path, 0, cx, cy, ww, hh)

            # Grid JSON (bit matrix)
            grid_json = self.out_dir/"labels"/"grid"/f"synth_{idx:05d}.json"
            write_grid_json(grid_json, size, M.tolist(), decode_gt=None)

            index.append({
                "image_id": f"synth_{idx:05d}",
                "rect_path": str(rect_path.relative_to(self.out_dir)),
                "scene_path": str(scene_path.relative_to(self.out_dir)),
                "grid_size": size,
                "polarity": polarity
            })

        # write metadata
        import csv
        meta_path = self.out_dir/"metadata.csv"
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id","rect_path","scene_path","grid_size","polarity"])
            writer.writeheader()
            writer.writerows(index)

        return {"count": len(index), "out_dir": str(self.out_dir)}
