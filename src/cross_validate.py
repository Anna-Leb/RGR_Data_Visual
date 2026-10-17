"""Обучение метода C и честная оценка всех методов по ручной разметке.

python src/cross_validate.py data/raw data/annotations [--merge-k 0.3 0.45 0.6]   (из корня проекта)

Leave-one-out: для каждого из N размеченных снимков модель обучается на
остальных N-1 и проверяется на нём (снимок, на котором меряем, модель не видела).
Результаты:
  results/cv_metrics.csv   — метрики по снимкам, методам и merge_k
  results/cv_summary.csv   — средние по методам
  models/rf_boundary.joblib — итоговая модель, обученная на всех N снимках
"""
import argparse, csv, glob
from pathlib import Path
import numpy as np
from sem_io import load_sem, read_magnifications
import pipeline, ml_boundary
from evaluate import evaluate, load_gt, annotated_region

ap = argparse.ArgumentParser()
ap.add_argument("images"); ap.add_argument("masks")
ap.add_argument("--merge-k", type=float, nargs="+", default=[0.45])
ap.add_argument("--n-per-image", type=int, default=40000)
ap.add_argument("--out", default="results"); ap.add_argument("--models", default="models")
a = ap.parse_args()

img_dir, mask_dir = Path(a.images), Path(a.masks)
mags = {}
if (img_dir / "magnifications.csv").exists():
    mags = read_magnifications(img_dir / "magnifications.csv")

# пары снимок–маска: маска называется как снимок (+ возможный суффикс _mask/_gt)
data = []
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
for m in sorted(p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT):
    stem = m.stem
    for suf in ("_mask", "_gt", "_label", "_labels", "-mask"):
        stem = stem.removesuffix(suf)
    cand = [p for p in img_dir.iterdir() if p.stem == stem and p.suffix.lower() in IMG_EXT]
    if not cand:
        print("нет снимка для маски", m.name); continue
    d = load_sem(cand[0], mag_kx=mags.get(stem))
    gt = load_gt(m, d["image"].shape)
    valid = annotated_region(gt)
    if gt.shape != d["image"].shape:
        raise SystemExit(f"{m.name}: размер маски {gt.shape} != рабочей области {d['image'].shape}")
    L = pipeline.estimate_scale(d["image"]); p = pipeline.params_from_scale(L)
    data.append({"name": stem, "d": d, "gt": gt, "valid": valid, "L": L,
                 "pre": pipeline.preprocess(d["image"], p)})
    print(f"{stem}: {gt.max()} эталонных зёрен, размечено {valid.mean():.0%} площади, L = {L:.1f} px")

rows = []
for i, test in enumerate(data):
    train = [(x["pre"], x["gt"], x["L"], x["valid"]) for j, x in enumerate(data) if j != i]
    clf = ml_boundary.train(train, n_per_image=a.n_per_image) if train else None
    for k in a.merge_k:
        out = pipeline.run(test["d"]["image"], test["d"]["pixel_nm"], merge_k=k, rf_model=clf)
        for meth in out["methods"]:
            for stage, lab in (("raw", out[meth]["labels"]), ("post", out[meth]["post"])):
                r = evaluate(test["gt"], lab, valid=test["valid"])
                rows.append({"image": test["name"], "method": meth, "stage": stage,
                             "merge_k": k, **r})
        e = [r for r in rows if r["image"] == test["name"] and r["method"] == "E"
             and r["stage"] == "post" and r["merge_k"] == k][0]
        print(f"[{test['name']}] merge_k={k}: E  F1={e['f1']:.3f}  BF1={e['boundary_f1']:.3f}  "
              f"split={e['split_gt']} merge={e['merged_pred']}")

Path(a.out).mkdir(parents=True, exist_ok=True)
with open(Path(a.out) / "cv_metrics.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys(), delimiter=";"); w.writeheader(); w.writerows(rows)
keys = ["dice", "f1", "pq", "boundary_f1", "split_gt", "merged_pred", "d_median_rel_err"]
summ = []
for meth in sorted({r["method"] for r in rows}):
    for stage in ("raw", "post"):
        for k in a.merge_k:
            sel = [r for r in rows if r["method"] == meth and r["stage"] == stage and r["merge_k"] == k]
            summ.append({"method": meth, "stage": stage, "merge_k": k,
                         **{q: round(float(np.mean([r[q] for r in sel])), 4) for q in keys}})
with open(Path(a.out) / "cv_summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=summ[0].keys(), delimiter=";"); w.writeheader(); w.writerows(summ)
print("\nСредние (post):")
for s in summ:
    if s["stage"] == "post":
        print(f"  {s['method']} k={s['merge_k']}: F1={s['f1']:.3f} PQ={s['pq']:.3f} "
              f"BF1={s['boundary_f1']:.3f} split={s['split_gt']:.1f} merge={s['merged_pred']:.1f}")

Path(a.models).mkdir(parents=True, exist_ok=True)
final = ml_boundary.train([(x["pre"], x["gt"], x["L"], x["valid"]) for x in data], n_per_image=a.n_per_image)
ml_boundary.save(final, Path(a.models) / "rf_boundary.joblib")
print("модель сохранена:", Path(a.models) / "rf_boundary.joblib")