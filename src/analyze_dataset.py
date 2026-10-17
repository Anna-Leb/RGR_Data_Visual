"""Шаг 1: технический анализ набора SEM-снимков -> results/dataset_analysis.csv"""
import sys, csv
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation
from sem_io import load_sem


def noise_sigma(img):
    """Оценка σ шума методом Immerkaer (1996), в долях от [0,1]."""
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    c = ndi.convolve(img.astype(float), M)[1:-1, 1:-1]
    return np.sqrt(np.pi / 2) * np.abs(c).sum() / (6 * c.size)


def rough_grain_size(img, h=0.03):
    """Грубая оценка числа зёрен: максимумы яркости + водораздел."""
    sm = filters.gaussian(img, 1.5 if img.shape[1] > 500 else 0.7)
    markers = measure.label(morphology.h_maxima(sm, h))
    lab = segmentation.watershed(-sm, markers)
    n = lab.max()
    return n, np.sqrt(4 * img.size / n / np.pi)


def analyze(path, mags):
    d = load_sem(path, mag_kx=mags.get(Path(path).stem))
    w = d["image"]
    sig = noise_sigma(w)
    bg = filters.gaussian(w, w.shape[1] / 10)
    n, d_px = rough_grain_size(w)
    px = d["pixel_nm"]
    return {
        "image": d["name"],
        "format": Path(path).suffix.lower(),
        "width": w.shape[1], "work_height": w.shape[0],
        "nm_per_px": round(px, 3) if px else "",
        "scale_source": d["scale_source"],
        "gray_levels": len(np.unique(np.round(w * 255))),
        "noise_sigma_255": round(sig * 255, 2),
        "snr": round(w.std() / sig, 1),
        "illum_nonuniform_%": round(100 * (np.percentile(bg, 99) - np.percentile(bg, 1)) / w.mean(), 1),
        "rough_grain_count": n,
        "rough_grain_d_px": round(d_px, 1),
        "rough_grain_d_nm": round(d_px * px) if px else "",
    }


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    mags = {}
    mag_file = folder / "magnifications.csv"
    if mag_file.exists():
        for row in csv.DictReader(open(mag_file, encoding="utf-8")):
            mags[row["image"]] = float(row["mag_kx"])
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".bmp"))
    rows = [analyze(p, mags) for p in files]
    out = Path("results"); out.mkdir(exist_ok=True)
    with open(out / "dataset_analysis.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=";")
        wr.writeheader(); wr.writerows(rows)
    for r in rows:
        print(r)
