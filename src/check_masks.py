"""Проверка масок: как программа их поняла.

python src/check_masks.py data/raw data/annotations
-> results/mask_check/<снимок>.png : снимок + контуры зёрен из маски
"""
import sys, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage import segmentation, color
from sem_io import load_sem
from evaluate import load_gt, annotated_region

img_dir, mask_dir = Path(sys.argv[1]), Path(sys.argv[2])
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
out = Path("results/mask_check"); out.mkdir(parents=True, exist_ok=True)
for m in sorted(p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT):
    stem = m.stem
    for suf in ("_mask", "_gt", "_label", "_labels", "-mask"):
        stem = stem.removesuffix(suf)
    im = Image.open(m)
    info = f"{m.name}: режим {im.mode}, размер {im.size[0]}x{im.size[1]}"
    cand = [p for p in img_dir.iterdir() if p.stem == stem and p.suffix.lower() in IMG_EXT]
    if not cand:
        print(info, "-> НЕТ СНИМКА с именем", stem); continue
    d = load_sem(cand[0])
    gt = load_gt(m, d["image"].shape)
    valid = annotated_region(gt)
    ok = gt.shape == d["image"].shape
    print(info, f"-> зёрен: {gt.max()}, размечено {valid.mean():.0%} рабочей области",
          "OK" if ok else f"РАЗМЕР НЕ СОВПАДАЕТ: маска {gt.shape}, снимок {d['image'].shape}")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(color.label2rgb(np.maximum(gt, 0), bg_label=0)); ax[0].set_title(f"маска: {gt.max()} зёрен")
    if ok:
        rgb = segmentation.mark_boundaries(d["image"], np.maximum(gt, 0), color=(1, 0, 0))
        rgb[~valid] = rgb[~valid] * 0.45          # неразмеченное затемнено
        ax[1].imshow(rgb)
    ax[1].set_title("контуры маски на снимке (тёмное = не размечено)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(out / f"{stem}.png", dpi=80); plt.close()
print("Картинки для проверки:", out)
