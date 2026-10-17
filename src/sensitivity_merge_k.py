"""Чувствительность результата к merge_k (порогу слияния соседних зёрен).

    python src/sensitivity_merge_k.py data/raw [--k 0.2 0.3 0.45 0.6 0.8 1.0]
                                      [--model models/rf_boundary.joblib] [--out results]

Работает БЕЗ эталонных масок, поэтому годится для любых снимков, в том числе
новых. Тяжёлая часть конвейера (предобработка, методы A/B/C, ансамбль) считается
для каждого снимка один раз, затем повторяется только постобработка для каждого
merge_k. Для каждого снимка и метода сохраняется число целых (не краевых) зёрен
и медианный эквивалентный диаметр.

Результаты:
  results/sensitivity_merge_k.csv  — все значения
  results/sensitivity_merge_k.png  — число зёрен и медианный диаметр от merge_k
                                     (метод E, относительно значения по умолчанию)
Для метрик качества по эталону при разных merge_k используйте
cross_validate.py --merge-k 0.3 0.45 0.6 ...
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import filters

import pipeline
from sem_io import load_sem, read_magnifications, IMG_EXT

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--k", type=float, nargs="+", default=[0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
ap.add_argument("--model", default=None, help="models/rf_boundary.joblib — добавляет метод C")
ap.add_argument("--out", default="results")
a = ap.parse_args()

folder = Path(a.folder)
mags = {}
if (folder / "magnifications.csv").exists():
    mags = read_magnifications(folder / "magnifications.csv")
clf = None
if a.model:
    import ml_boundary
    clf = ml_boundary.load(a.model)

files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
rows = []
for f in files:
    d = load_sem(f, mag_kx=mags.get(f.stem))
    img, px = d["image"], d["pixel_nm"]
    out = pipeline.run(img, px, rf_model=clf)          # единственный дорогой вызов на снимок
    p = out["params"]
    flat = filters.gaussian(img / np.maximum(filters.gaussian(img, p["bg_sigma"]), 1e-3), 1)
    consensus = out["E"]["consensus"]
    for key in out["methods"]:
        ks = sorted(set(a.k) | {pipeline.DEFAULT_MERGE_K[key]})
        for k in ks:
            lab, voids, edge_ids = pipeline.postprocess(out[key]["labels"], out[key]["edge"],
                                                        flat, p, k, ref_mask=consensus)
            area = np.bincount(lab.ravel())
            inner_ids = [i for i in np.flatnonzero(area) if i != 0 and i not in edge_ids]
            d_px = np.sqrt(4 * area[inner_ids] / np.pi) if inner_ids else np.array([np.nan])
            rows.append({
                "image": f.stem, "method": key, "merge_k": k,
                "is_default": int(k == pipeline.DEFAULT_MERGE_K[key]),
                "grains_total": int((area[1:] > 0).sum()), "grains_inner": len(inner_ids),
                "median_d_px": round(float(np.median(d_px)), 2),
                "median_d_nm": round(float(np.median(d_px)) * px, 1) if px else "",
            })
    print(f"{f.stem}: готово")

Path(a.out).mkdir(parents=True, exist_ok=True)
with open(Path(a.out) / "sensitivity_merge_k.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys(), delimiter=";")
    w.writeheader(); w.writerows(rows)

# --- сводка: насколько результат плывёт при сдвиге merge_k на один шаг сетки от значения по умолчанию
print("\nИзменение числа целых зёрен при сдвиге merge_k к ближайшим соседям по сетке")
print("(относительно значения по умолчанию; чем меньше, тем устойчивее):")
for img_name in sorted({r["image"] for r in rows}):
    for key in sorted({r["method"] for r in rows if r["image"] == img_name}):
        sel = sorted((r for r in rows if r["image"] == img_name and r["method"] == key),
                     key=lambda r: r["merge_k"])
        i0 = next(i for i, r in enumerate(sel) if r["is_default"])
        n0 = sel[i0]["grains_inner"]
        lo = sel[i0 - 1] if i0 > 0 else None
        hi = sel[i0 + 1] if i0 < len(sel) - 1 else None
        s = "  ".join(f"k={r['merge_k']:.2f}: {100 * (r['grains_inner'] - n0) / max(n0, 1):+.0f}%"
                      for r in (lo, hi) if r)
        print(f"  {img_name:10s} {key}  (k={sel[i0]['merge_k']:.2f}, {n0} зёрен)  {s}")

# --- график для метода E
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for img_name in sorted({r["image"] for r in rows}):
    sel = sorted((r for r in rows if r["image"] == img_name and r["method"] == "E"),
                 key=lambda r: r["merge_k"])
    base = next(r for r in sel if r["is_default"])
    ks = [r["merge_k"] for r in sel]
    ax[0].plot(ks, [r["grains_inner"] / max(base["grains_inner"], 1) for r in sel], "o-", label=img_name)
    ax[1].plot(ks, [r["median_d_px"] / base["median_d_px"] for r in sel], "o-", label=img_name)
for x, t in zip(ax, ("Число целых зёрен", "Медианный диаметр")):
    x.axvline(pipeline.DEFAULT_MERGE_K["E"], color="gray", ls="--", lw=1)
    x.axhline(1, color="gray", lw=0.6)
    x.set_xlabel("merge_k (метод E)"); x.set_ylabel(t + ", отн. значения по умолчанию")
    x.grid(alpha=0.3)
ax[0].legend(fontsize=8)
fig.suptitle("Чувствительность к порогу слияния merge_k (пунктир — значение по умолчанию)")
fig.tight_layout(); fig.savefig(Path(a.out) / "sensitivity_merge_k.png", dpi=110)
print("\nСохранено:", Path(a.out) / "sensitivity_merge_k.csv", "и", Path(a.out) / "sensitivity_merge_k.png")