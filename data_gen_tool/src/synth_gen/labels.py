
import json
from pathlib import Path

def write_yolo_label(path_txt: Path, cls_id: int, cx, cy, w, h):
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(path_txt, "w") as f:
        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def write_grid_json(path_json: Path, grid_size: int, bit_matrix, decode_gt=None):
    path_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "grid_size_known": grid_size,
        "decode_gt": decode_gt,
        "bit_matrix": bit_matrix
    }
    with open(path_json, "w") as f:
        json.dump(data, f)
