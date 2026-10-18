"""Логика приложения без графического интерфейса (её можно тестировать отдельно).

compute()       — полный расчёт одного снимка всеми методами
final_result()  — разметка выбранного метода + ручные правки + характеристики
export_one()    — сохранение всех файлов результата одного снимка
stages_figure() — рисунок с 6 обязательными стадиями (matplotlib Figure)
"""
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage import segmentation, color, measure

import pipeline
import editing as ed
from sem_io import load_sem, read_magnifications as _read_mags

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "rf_boundary.joblib"
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
METHOD_NAMES = {"A": "A — гребни (Sato) + водораздел",
                "B": "B — яркость зерна + водораздел",
                "C": "C — Random Forest (обучаемый)",
                "E": "E — ансамбль (итоговый)"}

_model = None


def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        import ml_boundary
        _model = ml_boundary.load(MODEL_PATH)
    return _model


def read_magnifications(path):
    """magnifications.csv -> {имя: Mag, K X}; терпит BOM, ';', десятичную запятую (см. sem_io)."""
    return _read_mags(path)


def compute(path, mag_kx=None, pixel_nm=None, use_model=True, merge_k=None, cut_row=None):
    d = load_sem(path, mag_kx=mag_kx, pixel_nm=pixel_nm, cut_row=cut_row)
    clf = get_model() if use_model else None
    out = pipeline.run(d["image"], d["pixel_nm"], merge_k=merge_k, rf_model=clf)
    res = {"image": d["image"], "pixel_nm": d["pixel_nm"], "scale_source": d["scale_source"],
           "cut_row": d["cut_row"], "cut_source": d["cut_source"], "warnings": d["warnings"],
           "full_height": d["full"].shape[0], "params": out["params"], "pre": out["pre"],
           "methods": out["methods"], "agreement": out["E"]["boundary_agreement"]}
    for k in out["methods"]:
        res[k] = {key: out[k][key] for key in ("edge", "labels", "post", "voids", "edge_ids", "grains")}
    return res


def final_result(R, method, ops):
    """Разметка метода + ручные правки -> labels, классы, карта классов, таблица зёрен."""
    M = R[method]
    lab, cls, log = ed.apply_ops(M["post"], M["edge"], ops)
    eids = ed.edge_ids_of(lab)
    cm = ed.class_map(lab, cls, M["voids"], eids)
    grains = pipeline.measure_grains(lab, M["edge"], R["pre"], eids, R["pixel_nm"])
    for g in grains:
        g["class"] = cls.get(g["id"], "edge" if g["id"] in eids else "grain")
    return {"lab": lab, "cls": cls, "cm": cm, "grains": grains, "log": log, "method": method}


def grain_stats(grains, pixel_nm):
    inner = [g for g in grains if g["class"] == "grain"]
    if pixel_nm:
        d = np.array([g["eq_diameter_nm"] for g in inner])
    else:
        d = np.array([np.sqrt(4 * g["area_px"] / np.pi) for g in inner])
    return {
        "n": len(inner), "unit": "нм" if pixel_nm else "пикс.",
        "d": d,
        "median": float(np.median(d)) if len(d) else float("nan"),
        "mean": float(d.mean()) if len(d) else float("nan"),
        "std": float(d.std()) if len(d) else float("nan"),
        "ecc": float(np.mean([g["eccentricity"] for g in inner])) if inner else float("nan"),
        "circ": np.array([g["circularity"] for g in inner]),
    }


def stages_figure(R, method, fig=None):
    """6 обязательных стадий: Original … Overlay."""
    M = R[method]
    img = R["image"]
    fig = fig or Figure(figsize=(13, 7.5))
    fig.clear()
    ax = fig.subplots(2, 3)
    ax = ax.ravel()
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("1. Оригинал")
    ax[1].imshow(R["pre"], cmap="gray"); ax[1].set_title("2. Предобработка")
    ax[2].imshow(M["edge"], cmap="magma")
    ax[2].set_title("3. Карта вероятностей (RF)" if method == "C" else "3. Карта границ")
    ax[3].imshow(segmentation.mark_boundaries(img, M["labels"], color=(1, 0, 0)))
    ax[3].set_title(f"4. Первичная сегментация ({int(M['labels'].max())})")
    ax[4].imshow(color.label2rgb(M["post"], bg_label=0, alpha=1))
    ax[4].set_title(f"5. Постобработка ({int(M['post'].max())})")
    cm = ed.class_map(M["post"], {}, M["voids"], M["edge_ids"])
    ax[5].imshow(ed.render(img, M["post"], cm)); ax[5].set_title("6. Накладка на оригинал")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{METHOD_NAMES[method]}   L = {R['params']['L']:.1f} пикс.")
    fig.tight_layout()
    return fig


def export_one(out_dir, stem, R, fin, ops):
    """Сохраняет все результаты одного снимка в папку out_dir/stem."""
    od = Path(out_dir) / stem
    od.mkdir(parents=True, exist_ok=True)
    files = {
        "labels.png": ed.png_bytes(fin["lab"].astype(np.uint16)),
        "classes.png": ed.png_bytes(fin["cm"]),
        "classes_color.png": ed.png_bytes(ed.class_map_rgb(fin["cm"])),
        "overlay.png": ed.png_bytes(ed.render(R["image"], fin["lab"], fin["cm"])),
        "grains.csv": ed.csv_bytes(fin["grains"]),
        "corrections.json": ed.corrections_json(stem, fin["method"], ops,
                                                {"pixel_nm": R["pixel_nm"]}).encode("utf-8"),
    }
    for name, data in files.items():
        (od / name).write_bytes(data)
    for m in R["methods"]:
        fig = stages_figure(R, m)
        FigureCanvasAgg(fig)
        fig.savefig(od / f"stages_{m}.png", dpi=80)
    return od


def summary_row(stem, R, fin):
    st = grain_stats(fin["grains"], R["pixel_nm"])
    return {
        "снимок": stem, "метод": fin["method"],
        "нм/пикс": round(R["pixel_nm"], 3) if R["pixel_nm"] else "",
        "зёрен": st["n"],
        "пор": int(measure.label(fin["cm"] == ed.CLS_ID["void"]).max()),
        "единицы d": st["unit"],
        "медиана d": round(st["median"], 1),
        "среднее d": round(st["mean"], 1),
        "СКО d": round(st["std"], 1),
        "согласие методов": round(R["agreement"], 3),
    }