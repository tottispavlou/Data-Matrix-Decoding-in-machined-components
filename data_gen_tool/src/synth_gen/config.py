
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SynthConfig:
    out_dir: str = "example_output"
    n_samples: int = 20
    grid_sizes: tuple = (14, 16)
    cell_px: int = 24                 # pixels per cell (rectified crop)
    dot_radius_frac: float = 0.35     # dot radius as fraction of cell size
    dot_radius_jitter: float = 0.08   # random jitter on radius
    polarity_prob: float = 0.5        # probability of 'light-on-dark' vs 'dark-on-light'
    overlap_prob: float = 0.1         # chance of slight overlap deformation

    # Background textures
    metal_textures_dir: Optional[str] = None   # folder with metal tiles (jpg/png)
    use_real_textures_prob: float = 0.7        # chance to use a real texture (else procedural)
    texture_blend: float = 0.5                 # blend real texture with procedural base [0..1]
    orient_blur_prob: float = 0.7              # chance to apply oriented blur to simulate brushing
    orient_blur_kernel: int = 31               # size of directional blur kernel
    orient_deg_range: Tuple[float,float] = (-20.0, 20.0)  # brushing orientation range (deg)

    # Glare (shining) and scratches
    glare_prob: float = 0.7
    glare_layers: Tuple[int,int] = (1, 3)      # min/max glare stripes to add
    glare_strength_range: Tuple[float,float] = (0.25, 0.85)
    glare_width_range: Tuple[float,float] = (8.0, 28.0)
    scratch_prob: float = 0.5
    scratch_count_range: Tuple[int,int] = (20, 80)
    scratch_length_range: Tuple[int,int] = (15, 60)
    scratch_thickness_range: Tuple[int,int] = (1, 2)

    # Perspective & output
    perspective_prob: float = 0.6
    scene_scale: float = 2.0
    jpeg_quality: int = 95
    seed: int = 42
