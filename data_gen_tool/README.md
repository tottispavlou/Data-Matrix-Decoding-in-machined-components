
# Dot-Peen Synthetic Data Generator (MVP)

This is a minimal generator for **dot-peen-like** Data Matrix images to support grid-first pipelines.
It **does not** implement true ECC200 encoding; instead it builds a **faux** Data Matrix with a correct
finder L and alternating timing borders, and a random interior bit pattern. That is sufficient for
training **grid detection, rectification, and per-cell classification**.

## Features
- Rectified crop rendering of a 14×14 or 16×16 bit matrix as **embossed/indented dots**.
- Procedural **brushed metal** background, optional **glare** and **scratches**.
- Optional **perspective warp** to create an unrectified full scene.
- Outputs **YOLO labels** for the scene, and **bit-matrix JSON** for the crop.
- `metadata.csv` index for quick inspection.

## Layout
```
src/synth_gen/
  config.py          # SynthConfig dataclass
  ecc200_like.py     # ECC200-like (NOT real ECC) matrix generator
  metal.py           # background texture and artifacts
  dotpeen.py         # dot rendering on rectified grid
  geom.py            # perspective warp helpers
  labels.py          # writers for YOLO txt and grid JSON
  generator.py       # SynthGenerator orchestrator
scripts/
  gen_synth.py       # CLI entry
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

## Usage
```bash
python scripts/gen_synth.py --out_dir example_output --n 50 --cell_px 24
```

Outputs will be in `example_output/`:
- `images/rect/*.png`   - rectified crops
- `images/scene/*.jpg`  - unrectified scenes
- `labels/yolo/*.txt`   - YOLO labels (class 0)
- `labels/grid/*.json`  - grid metadata with bit-matrix
- `metadata.csv`

## Notes
- For **true libdmtx decodability**, you'd need a real ECC200 encoder to build the interior data region.
  For our **grid-first** pipeline and per-cell classifier training, faux interior bits are fine.
- Tweak parameters in `SynthConfig` to match your real audit stats (glare, blur, scale, etc.).


## Use real metal textures (recommended)
1. After you have YOLO boxes for your real images, harvest background tiles:
```bash
python scripts/harvest_textures.py --images_dir /path/to/your/images \
  --labels_dir /path/to/your/yolo_labels \
  --out_dir data/textures/metal --tile_size 256 --per_image 6
```

2. Point the generator at that folder:
```bash
python scripts/gen_synth.py --out_dir example_output --n 100
```
…and in `src/synth_gen/config.py` set:
```python
metal_textures_dir = "data/textures/metal"
use_real_textures_prob = 0.7   # 0..1, how often to use a real texture
texture_blend = 0.5            # blend with procedural base
```
The generator will **blend** real tiles with procedural brushed metal and add **multi-stripe glare** and **scratches**. Tune `glare_*` and brushing (`orient_*`) to match your audit.

