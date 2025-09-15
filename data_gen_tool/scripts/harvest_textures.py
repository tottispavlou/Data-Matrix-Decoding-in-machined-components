
#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import cv2
import numpy as np

def parse_yolo_txt(txt_path, img_w, img_h):
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5 and len(parts) != 6:
                # allow 'class cx cy w h [conf]' format
                continue
            cls = int(parts[0]); cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            bx = int((cx - w/2) * img_w)
            by = int((cy - h/2) * img_h)
            bw = int(w * img_w)
            bh = int(h * img_h)
            boxes.append((bx, by, bw, bh))
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Folder with images (jpg/png)")
    ap.add_argument("--labels_dir", required=True, help="Folder with YOLO txt labels (same basename)")
    ap.add_argument("--out_dir", required=True, help="Folder to write cropped metal textures")
    ap.add_argument("--tile_size", type=int, default=256, help="Texture tile size to extract")
    ap.add_argument("--per_image", type=int, default=6, help="Max tiles per image")
    ap.add_argument("--margin", type=float, default=0.15, help="Margin around DMC to avoid when sampling")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

    count = 0
    for img_path in Path(args.images_dir).rglob('*'):
        if img_path.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        label_path = Path(args.labels_dir) / (img_path.stem + ".txt")
        boxes = parse_yolo_txt(label_path, w, h)

        # build a mask to AVOID DMC region (with margin)
        mask = np.zeros((h, w), np.uint8)
        for (bx, by, bw, bh) in boxes:
            mx0 = max(0, int(bx - args.margin * bw))
            my0 = max(0, int(by - args.margin * bh))
            mx1 = min(w, int(bx + bw + args.margin * bw))
            my1 = min(h, int(by + bh + args.margin * bh))
            mask[my0:my1, mx0:mx1] = 255

        # sample tiles away from mask
        taken = 0
        tries = 0
        while taken < args.per_image and tries < 50:
            tries += 1
            if h < args.tile_size or w < args.tile_size:
                break
            y0 = np.random.randint(0, h - args.tile_size + 1)
            x0 = np.random.randint(0, w - args.tile_size + 1)
            if mask[y0:y0+args.tile_size, x0:x0+args.tile_size].max() == 0:
                tile = img[y0:y0+args.tile_size, x0:x0+args.tile_size]
                out_path = out / f"{img_path.stem}_{taken:02d}.png"
                cv2.imwrite(str(out_path), tile, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                taken += 1
                count += 1

    print(f"Extracted {count} texture tiles to {out}")

if __name__ == "__main__":
    main()
