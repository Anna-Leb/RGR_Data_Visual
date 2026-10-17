"""Пакетная обработка (из корня проекта):
   python src/run_batch.py data/raw [--model models/rf_boundary.joblib] [--merge-k 0.6]

Для каждого снимка и каждого метода (A — гребни, B — яркость,
C — Random Forest, если задан --model, E — ансамбль):
  results/<снимок>/panel_<метод>.png   — 6 обязательных стадий
  results/<снимок>/grains_<метод>.csv  — характеристики зёрен
  results/<снимок>/labels_<метод>.png  — маска экземпляров (16 бит)
и сводная results/summary.csv.
"""
import argparse, csv, time
from pathlib import Path
import numpy as np
from PIL import Image
from sem_io import load_sem, read_magnifications
import pipeline, viz

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--merge-k", type=float, default=None,
                help="по умолчанию — значения, подобранные кросс-валидацией")
ap.add_argument("--out", default="results")
ap.add_argument("--model", default=None, help="models/rf_boundary.joblib — включает метод C")
a = ap.parse_args()

folder = Path(a.folder)
mags = {}
if (folder / "magnifications.csv").exists():
    mags = read_magnifications(folder / "magnifications.csv")
    print(f"magnifications.csv: распознано увеличений: {len(mags)}")
    if not mags:
        print("  ВНИМАНИЕ: файл не распознан — нужны колонки image и mag_kx")
clf = None
if a.model:
    import ml_boundary
    clf = ml_boundary.load(a.model)
files = sorted(p for p in folder.iterdir()
               if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"))
summary = []
for f in files:
    d = load_sem(f, mag_kx=mags.get(f.stem))
    for w in d["warnings"]:
        print(f"{f.stem}: ВНИМАНИЕ: {w}")
    t = time.time()
    out = pipeline.run(d["image"], d["pixel_nm"], merge_k=a.merge_k, rf_model=clf)
    dt = time.time() - t
    od = Path(a.out) / f.stem; od.mkdir(parents=True, exist_ok=True)
    for key in out["methods"]:
        M = out[key]
        viz.panel(f.stem, d["image"], out, key, od / f"panel_{key}.png")
        Image.fromarray(M["post"].astype(np.uint16)).save(od / f"labels_{key}.png")
        g = M["grains"]
        if g:
            with open(od / f"grains_{key}.csv", "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=g[0].keys(), delimiter=";"); w.writeheader(); w.writerows(g)
        inner = [x for x in g if not x["touches_border"]]
        dnm = [x["eq_diameter_nm"] for x in inner] if d["pixel_nm"] else []
        dpx = [float(np.sqrt(4 * x["area_px"] / np.pi)) for x in inner]   # всегда, даже без масштаба
        summary.append({
            "image": f.stem, "method": key,
            "nm_per_px": round(d["pixel_nm"], 3) if d["pixel_nm"] else "",
            "L_px": round(out["params"]["L"], 1),
            "raw_objects": int(M["labels"].max()), "grains": int(M["post"].max()),
            "inner_grains": len(inner),
            "median_d_nm": round(float(np.median(dnm)), 1) if dnm else "",
            "mean_d_nm": round(float(np.mean(dnm)), 1) if dnm else "",
            "std_d_nm": round(float(np.std(dnm)), 1) if dnm else "",
            "median_d_px": round(float(np.median(dpx)), 1) if dpx else "",
            "mean_d_px": round(float(np.mean(dpx)), 1) if dpx else "",
            "std_d_px": round(float(np.std(dpx)), 1) if dpx else "",
            "voids": int(pipeline.measure.label(M["voids"]).max()),
            "boundary_agreement_AB": round(out["E"]["boundary_agreement"], 3),
            "time_s": round(dt, 1),
        })
    s = summary[-1]
    dmed = f"{s['median_d_nm']} nm" if s["median_d_nm"] != "" else f"{s['median_d_px']} px (масштаб не задан)"
    print(f"{f.stem}: E -> {s['inner_grains']} inner grains, median d = {dmed}, {dt:.1f} s")
with open(Path(a.out) / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=summary[0].keys(), delimiter=";"); w.writeheader(); w.writerows(summary)