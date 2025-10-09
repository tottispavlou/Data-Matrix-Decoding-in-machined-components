#!/usr/bin/env python3
"""
Generate synthetic dot-peen Data Matrix images by STAMPING REAL PATCHES per cell,
then composing on a metal background with glare/scratches. Outputs rectified crops,
full scenes, YOLO labels, and a grid JSON with the bit matrix.

Inputs (folders you create):
  --dot_dir   data/patches/dot       # REQUIRED: single-cell dot patches (grayscale ok)
  --space_dir data/patches/space     # OPTIONAL: single-cell empty patches
  --tex_dir   data/textures/metal    # OPTIONAL: metal textures to blend into background

Outputs:
  out/images/rect/*.png
  out/images/scene/*.jpg
  out/labels/yolo/*.txt     (class 0)
  out/labels/grid/*.json    (grid_size_known + bit_matrix)
  out/metadata.csv
"""

import os, argparse, json, csv, random, math
from pathlib import Path
import numpy as np
import cv2

# ---------------------- helpers ----------------------

def normalize_pts(pts, W, H):
    # pts: (4,2) float32 in px -> normalized [0,1]
    out = pts.copy().astype(np.float32)
    out[:, 0] /= float(W)
    out[:, 1] /= float(H)
    return out

def order_quad_clockwise(pts):
    # pts: (4,2). Return TL, TR, BR, BL (clockwise)
    # sort by y, then split top/bottom
    pts = np.array(pts, dtype=np.float32)   
    ys = pts[:,1]
    top_idx = np.argsort(ys)[:2]
    bot_idx = np.argsort(ys)[-2:]
    top = pts[top_idx]; bot = pts[bot_idx]
    # left/right by x within each row
    tl, tr = top[np.argsort(top[:,0])]
    bl, br = bot[np.argsort(bot[:,0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def write_obb_txt(path, cls_id, quad_norm):
    # quad_norm: (4,2) normalized, in order TL,TR,BR,BL
    x1,y1 = quad_norm[0]; x2,y2 = quad_norm[1]; x3,y3 = quad_norm[2]; x4,y4 = quad_norm[3]
    with open(path, "w") as f:
        f.write(f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")

def make_faux_datamatrix(size: int, payload_p: float = 0.5) -> np.ndarray:
    """
    Build a proper DataMatrix-like grid:
      - Finder L: left column + bottom row are solid 1s
      - Clocking: top row alternates 1,0,...; right column alternates 0,1,...
    """
    assert size % 2 == 0 and size >= 10, "size should be even (e.g., 14 or 16)"
    M = np.zeros((size, size), dtype=np.uint8)

    # Finder L
    M[:, 0] = 1
    M[-1, :] = 1

    # Clocking pattern
    for j in range(size):
        M[0, j] = 1 if (j % 2 == 0) else 0       # top: 1,0,1,0,...
    for i in range(size):
        M[i, -1] = 1 if (i % 2 == 1) else 0      # right: 0,1,0,1,...

    # Ensure corners are correct
    M[0, 0] = 1
    M[-1, 0] = 1
    M[-1, -1] = 1

    # Payload
    M[1:-1, 1:-1] = (np.random.rand(size-2, size-2) < payload_p).astype(np.uint8)
    return M


def load_imgs(folder: Path):
    if folder is None or not folder.exists(): return []
    imgs = []
    for p in folder.rglob("*"):
        if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is not None: imgs.append(im)
    return imgs

def feather_mask(h, w, feather_px=4):
    y, x = np.indices((h, w))
    cy, cx = (h-1)/2.0, (w-1)/2.0
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    r_max = min(cx, cy)
    mask = (r <= r_max).astype(np.float32)
    if feather_px > 0:
        edge = np.clip((r_max - r)/max(1e-6, feather_px), 0, 1)
        mask = mask * 0.5 * (1 - np.cos(np.pi * np.clip(edge, 0, 1)))
    return np.clip(mask, 0, 1)

def rotate_resize(patch, cell_px, rot_deg):
    h, w = patch.shape
    s = max(h, w)
    pad_y = (s-h)//2
    pad_x = (s-w)//2
    sq = cv2.copyMakeBorder(patch, pad_y, s-h-pad_y, pad_x, s-w-pad_x, cv2.BORDER_REFLECT_101)
    M = cv2.getRotationMatrix2D((s/2, s/2), rot_deg, 1.0)
    rot = cv2.warpAffine(sq, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return cv2.resize(rot, (cell_px, cell_px), interpolation=cv2.INTER_AREA)

def jitter_brightness_contrast(pf, b_jit=0.08, c_jit=0.12):
    # pf in [0,1]; (p-0.5)*(1+c)+0.5 then +b
    out = (pf - 0.5) * (1.0 + np.random.uniform(-c_jit, c_jit)) + 0.5
    out = np.clip(out + np.random.uniform(-b_jit, b_jit), 0, 1)
    return out

def dir_kernel(angle_deg: float, k: int):
    angle = np.deg2rad(angle_deg)
    kern = np.zeros((k, k), np.float32)
    cx = cy = k//2
    for t in np.linspace(-1, 1, 2*k):
        x = int(cx + t * (k//2) * np.cos(angle))
        y = int(cy + t * (k//2) * np.sin(angle))
        if 0 <= x < k and 0 <= y < k:
            kern[y, x] = 1.0
    s = kern.sum()
    if s > 0: kern /= s
    else: kern[cy, :] = 1.0 / k
    return kern

def procedural_metal(h, w, angle_deg=0.0, ksize=31):
    base = (np.random.rand(h, w) * 255).astype(np.uint8)
    kernel = dir_kernel(angle_deg, ksize)
    brushed = cv2.filter2D(base, -1, kernel)
    y, x = np.indices((h, w))
    cy, cx = h/2.0, w/2.0
    r = np.sqrt(((y - cy)/cy)**2 + ((x - cx)/cx)**2)
    vignette = np.clip(1.0 - 0.6 * r, 0, 1)
    return (brushed.astype(np.float32) * (0.7 + 0.3*vignette)).astype(np.uint8)

def sample_texture(tex_paths, H, W):
    if not tex_paths: return None
    path = random.choice(tex_paths)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    h, w = img.shape
    if h < H or w < W:
        scale = max(H/h, W/w)
        img = cv2.resize(img, (int(w*scale)+1, int(h*scale)+1), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape
    y0 = random.randint(0, h - H)
    x0 = random.randint(0, w - W)
    return img[y0:y0+H, x0:x0+W]

def compose_background(H, W, tex_paths=None, use_real_prob=0.7, blend=0.5, orient_blur=True, ksize=31, deg_range=(-20,20)):
    angle = np.random.uniform(*deg_range) if orient_blur else 0.0
    proc = procedural_metal(H, W, angle_deg=angle, ksize=ksize)
    tile = sample_texture(tex_paths, H, W) if (tex_paths and random.random() < use_real_prob) else None
    if tile is not None:
        tile_f = tile.astype(np.float32)/255.0
        proc_f = proc.astype(np.float32)/255.0
        bg = np.clip(blend*tile_f + (1-blend)*proc_f, 0, 1)
        return (bg*255).astype(np.uint8)
    return proc

def add_glare_multi(img, layers=(1,2), strength=(0.15,0.30), width=(8.0,30.0), base_angle=None, angle_jitter=30):
    h, w = img.shape
    out = img.astype(np.float32)/255.0
    if base_angle is None:
        base_angle = np.random.uniform(-30, 30)
    for _ in range(np.random.randint(layers[0], layers[1]+1)):
        ang = base_angle + np.random.uniform(-angle_jitter, angle_jitter)
        y, x = np.indices((h, w))
        theta = np.deg2rad(ang)
        xr = (x - w/2) * np.cos(theta) + (y - h/2) * np.sin(theta)
        wid = np.random.uniform(width[0], width[1])
        stripe = np.exp(-(xr**2)/(2*(wid**2)))
        stripe /= (stripe.max()+1e-6)
        s = np.random.uniform(strength[0], strength[1])
        out = np.clip(out + s*stripe, 0, 1)
    return (out*255).astype(np.uint8)

def add_scratches(img, count=(20,80), length=(15,60), thickness=(1,2)):
    out = img.copy()
    h, w = img.shape
    for _ in range(np.random.randint(count[0], count[1]+1)):
        x0 = np.random.randint(0, w); y0 = np.random.randint(0, h)
        ang = np.random.rand()*np.pi
        x1 = int(x0 + np.random.randint(length[0], length[1]+1)*np.cos(ang))
        y1 = int(y0 + np.random.randint(length[0], length[1]+1)*np.sin(ang))
        cv2.line(out, (x0,y0), (x1,y1), int(np.random.randint(180,255)), np.random.randint(thickness[0], thickness[1]+1))
    return out

def warp_perspective(img, strength):
    h, w = img.shape[:2]
    src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    jitter = strength * min(h,w)
    dst = src + np.random.uniform(-jitter, jitter, size=src.shape).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), H

def yolo_box_centered(img, pad=0):
    h, w = img.shape[:2]
    x = pad; y = pad; bw = w - 2*pad; bh = h - 2*pad
    return ( (x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h )

# ---------------------- core: render rectified from patches ----------------------

def render_rectified_with_patches(M, cell_px, dot_lib, space_lib=None,tex_paths=None,
                                  feather_px=4, rot_jitter=8.0,
                                  use_space_prob=0.15):
    h_cells, w_cells = M.shape
    H = h_cells*cell_px; W = w_cells*cell_px

    if tex_paths:
        base_tex = sample_texture(tex_paths, H, W)
        img = base_tex.astype(np.float32) / 255.0
    else:
        img = np.ones((H, W), dtype=np.float32) * 0.7


    mask = feather_mask(cell_px, cell_px, feather_px)
    centers = []
    for i in range(h_cells):
        for j in range(w_cells):
            cy = int((i+0.5)*cell_px); cx = int((j+0.5)*cell_px)
            centers.append((cy, cx))
            # choose patch
            if M[i,j] == 1:
                patch = random.choice(dot_lib).copy()
            else:
                if space_lib and random.random() < use_space_prob:
                    patch = random.choice(space_lib).copy()
                else:
                    continue
            rot = np.random.uniform(-rot_jitter, rot_jitter)
            patch = rotate_resize(patch, cell_px, rot)
            pf = patch.astype(np.float32)/255.0
            pf = jitter_brightness_contrast(pf)
            y0 = cy - cell_px//2; x0 = cx - cell_px//2
            y1 = y0 + cell_px;    x1 = x0 + cell_px
            if y0<0 or x0<0 or y1>H or x1>W: continue
            region = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = (1 - mask)*region + mask*pf
    img_u8 = (np.clip(img, 0, 1)*255).astype(np.uint8)
    img_u8 = cv2.GaussianBlur(img_u8, (3,3), 0.8)
    return img_u8, centers, cell_px

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="example_output_patched")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cell_px", type=int, default=24)
    ap.add_argument("--dot_dir", type=str, required=True, help="Folder with dot patches")
    ap.add_argument("--space_dir", type=str, default=None, help="Optional folder with empty patches")
    ap.add_argument("--tex_dir", type=str, default=None, help="Optional metal textures folder")
    ap.add_argument("--use_real_tex_prob", type=float, default=0.7)
    ap.add_argument("--tex_blend", type=float, default=0.5)
    ap.add_argument("--perspective_prob", type=float, default=0.6)
    ap.add_argument("--jpeg_q", type=int, default=95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid_sizes", type=int, nargs="+", default=[14, 16],
                    help="Allowed DataMatrix sizes")
    ap.add_argument("--scale_min", type=float, default=0.4)
    ap.add_argument("--scale_max", type=float, default=1.0)

    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    out = Path(args.out_dir)
    (out/"images"/"rect").mkdir(parents=True, exist_ok=True)
    (out/"images"/"scene").mkdir(parents=True, exist_ok=True)
    (out/"labels"/"yolo").mkdir(parents=True, exist_ok=True)
    (out/"labels"/"grid").mkdir(parents=True, exist_ok=True)

    dot_lib = load_imgs(Path(args.dot_dir))
    if len(dot_lib) == 0:
        raise SystemExit(f"[ERR] No dot patches found in {args.dot_dir}")
    space_lib = load_imgs(Path(args.space_dir)) if args.space_dir else None

    tex_paths = list(Path(args.tex_dir).rglob("*")) if args.tex_dir and Path(args.tex_dir).exists() else []

    index_rows = []
    for i in range(args.n):
        size = random.choice(args.grid_sizes)
        M = make_faux_datamatrix(size)

        # build rectified crop
        rect_img, _, _ = render_rectified_with_patches(M, args.cell_px, dot_lib, space_lib, tex_paths)

        # random scale
        scale = np.random.uniform(args.scale_min, args.scale_max)
        h, w = rect_img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        rect_img = cv2.resize(rect_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # build scene (2x rect size for margin)
        H, W = int(new_h * 2.0), int(new_w * 2.0)
        bg = compose_background(H, W, tex_paths, args.use_real_tex_prob, args.tex_blend,
                                orient_blur=True, ksize=31, deg_range=(-20,20))
        scene = bg.copy()

        # random placement
        H, W = scene.shape[:2]


        # Safe margin (e.g. 10% of background size)
        margin_x = int(0.1 * W)
        margin_y = int(0.1 * H)

        # Random placement inside safe area
        x0 = np.random.randint(margin_x, W - new_w - margin_x)
        y0 = np.random.randint(margin_y, H - new_h - margin_y)


        scene[y0:y0+new_h, x0:x0+new_w] = cv2.addWeighted(
            scene[y0:y0+new_h, x0:x0+new_w], 0.1, rect_img, 0.9, 0
        )

        # define quad from scaled size
        quad = np.array([
            [x0,        y0],
            [x0+new_w,  y0],
            [x0+new_w,  y0+new_h],
            [x0,        y0+new_h]
        ], dtype=np.float32)

        # add effects
        scene = add_glare_multi(scene)
        scene = add_scratches(scene)
        noise = np.random.normal(0, 3.0, size=scene.shape).astype(np.float32)
        scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # optional perspective warp
        Hmat = None
        if random.random() < args.perspective_prob:
            scene, Hmat = warp_perspective(scene, strength=np.random.uniform(0.05, 0.25))

        quad_warped = quad.copy()
        if Hmat is not None:
            quad_warped = cv2.perspectiveTransform(quad.reshape(1, -1, 2), Hmat).reshape(-1, 2)

        # save rect + scene
        rect_path  = out/"images"/"rect"/f"synth_{i:05d}.png"
        scene_path = out/"images"/"scene"/f"synth_{i:05d}.jpg"
        cv2.imwrite(str(rect_path), rect_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(scene_path), scene, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_q])

        # labels (OBB)
        quad_ord = order_quad_clockwise(quad_warped)
        quad_norm = normalize_pts(quad_ord, W, H)
        obb_path = out/"labels"/"yolo"/f"synth_{i:05d}.txt"
        write_obb_txt(obb_path, 0, quad_norm)

        index_rows.append({
            "image_id": f"synth_{i:05d}",
            "rect_path": str(rect_path.relative_to(out)),
            "scene_path": str(scene_path.relative_to(out)),
            "grid_size": size
        })

    with open(out/"metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id","rect_path","scene_path","grid_size"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"[OK] Generated {len(index_rows)} samples in {out}")

if __name__ == "__main__":
    main()

