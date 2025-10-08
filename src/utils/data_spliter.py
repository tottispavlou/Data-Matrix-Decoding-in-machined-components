import os, random, shutil
from pathlib import Path

random.seed(42)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

src_imgs = Path("preproc/divide")
src_labels = Path("raw/all/labels")

# make splits
splits = {"train":0.7, "val":0.15, "test":0.15}
all_imgs = [p for p in src_imgs.glob("*.jpg")] + [p for p in src_imgs.glob("*.png")]
random.shuffle(all_imgs)

n = len(all_imgs)
train_n = int(n * splits["train"])
val_n   = int(n * splits["val"])

train_imgs = all_imgs[:train_n]
val_imgs   = all_imgs[train_n:train_n+val_n]
test_imgs  = all_imgs[train_n+val_n:]

def copy_split(imgs, split):
    img_out = Path(f"raw/{split}/images")
    lbl_out = Path(f"raw/{split}/labels")
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img in imgs:
        lbl = src_labels / (img.stem + ".txt")
        shutil.copy(img, img_out/img.name)
        if lbl.exists():
            shutil.copy(lbl, lbl_out/lbl.name)

copy_split(train_imgs, "train")
copy_split(val_imgs, "val")
copy_split(test_imgs, "test")
