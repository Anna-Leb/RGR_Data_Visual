"""Сегментация зёрен на SEM-снимках.

Конвейер:
  0. load_sem          — рабочая область + масштаб (sem_io.py)
  1. estimate_scale    — средний размер зерна методом секущих -> все параметры
  2. preprocess        — выравнивание фона, шумоподавление, CLAHE
  3a. method_ridge     — водораздел по карте тёмных гребней (межзёренных границ)
  3b. method_intensity — водораздел по градиенту, маркеры = яркие центры зёрен
  4. ensemble          — согласованные маркеры двух методов + общая карта границ
  5. postprocess       — поры/трещины, слияние по слабой границе (RAG),
                         поглощение мелких фрагментов, краевые зёрна
  6. measure_grains    — характеристики в пикселях и нм
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import (filters, morphology, measure, segmentation, exposure,
                     feature, graph)


def remove_small(mask, size):
    """remove_small_objects, совместимый со старыми (min_size) и новыми (max_size) scikit-image."""
    try:
        return morphology.remove_small_objects(mask, max_size=size)
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=size + 1)


# ---------------------------------------------------------------- 1. масштаб
def estimate_scale(img):
    """Средняя длина секущей L (пикс.) по скелету тёмных гребней.

    L ≈ характерный размер зерна. От L считаются все параметры алгоритма,
    поэтому один и тот же код работает при зёрнах от 10 до 120 пикселей.
    """
    s = 2.0 * np.sqrt(img.shape[1] / 400)
    r = filters.sato(filters.gaussian(img, 0.5), sigmas=[s], black_ridges=True)
    b = r > filters.threshold_otsu(r)
    b = remove_small(b, 20)
    b = morphology.skeletonize(b)
    n = (np.diff(b.astype(np.int8), axis=1) == 1).sum() + \
        (np.diff(b.astype(np.int8), axis=0) == 1).sum()
    return 2 * b.size / max(n, 1)


def params_from_scale(L):
    return {
        "L": L,
        "bg_sigma": 3 * L,                    # фон — масштаб многих зёрен
        "clahe_kernel": int(max(16, 4 * L)),  # окно CLAHE ~ 4 зерна
        "ridge_sigmas": list(np.clip([L / 18, L / 11], 0.8, 6)),
        "min_dist": max(3, int(L / 3)),       # мин. расстояние между маркерами
        "smooth_int": max(1.0, L / 7),        # сглаживание для ярких центров
        "min_area": max(8, int((L / 4) ** 2)),  # меньше — точно фрагмент
    }


def noise_sigma(img):
    """σ шума по Immerkaer (1996)."""
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    c = ndi.convolve(img.astype(float), M)[1:-1, 1:-1]
    return np.sqrt(np.pi / 2) * np.abs(c).sum() / (6 * c.size)


# ---------------------------------------------------------- 2. предобработка
def preprocess(img, p):
    # 2.1 выравнивание освещения: деление на крупномасштабный фон
    bg = filters.gaussian(img, p["bg_sigma"])
    flat = img / np.maximum(bg, 1e-3) * img.mean()
    flat = np.clip((flat - np.percentile(flat, 0.5)) /
                   (np.percentile(flat, 99.5) - np.percentile(flat, 0.5)), 0, 1)
    # 2.2 шумоподавление non-local means: сохраняет границы
    sig = noise_sigma(flat) * 255
    u8 = (flat * 255).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(u8, None, h=float(max(3, 1.2 * sig)),
                                   templateWindowSize=5, searchWindowSize=15)
    den = den.astype(np.float32) / 255
    # 2.3 локальный контраст
    out = exposure.equalize_adapthist(den, kernel_size=p["clahe_kernel"],
                                      clip_limit=0.01)
    return out.astype(np.float32)


# ------------------------------------------------- 3a. метод «по границам»
def method_ridge(pre, p):
    """Межзёренная граница = тёмный гребень. Sato-фильтр даёт карту границ."""
    edge = filters.sato(pre, sigmas=p["ridge_sigmas"], black_ridges=True)
    edge = edge / (np.percentile(edge, 99.5) + 1e-9)
    edge = np.clip(edge, 0, 1)
    bnd = edge > filters.threshold_otsu(edge)
    bnd = remove_small(bnd, p["min_area"])
    raw = ~bnd                                           # «сырая» маска зёрен
    dist = ndi.distance_transform_edt(raw)
    pk = feature.peak_local_max(dist, min_distance=p["min_dist"],
                                threshold_abs=p["L"] / 10, labels=measure.label(raw))
    markers = np.zeros(pre.shape, np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(filters.gaussian(edge, 1), markers)
    return {"edge": edge, "raw": raw, "markers": markers, "labels": lab}


# --------------------------------------------- 3b. метод «по яркости зерна»
def method_intensity(pre, p):
    """Центр зерна ярче его краёв (SEM-контраст рельефа, детектор InLens).

    Маркеры — локальные максимумы сглаженной яркости; карта границ —
    морфологический black top-hat (тёмные долины уже ~L/3). Водораздел идёт
    по долинам яркости, т.е. по центру границы, а не по её склону, как было бы
    на карте градиента.
    """
    sm = filters.gaussian(pre, p["smooth_int"])
    r = max(2, int(p["L"] / 3))
    th = morphology.black_tophat(filters.gaussian(pre, 1), morphology.disk(r))
    edge = np.clip(th / (np.percentile(th, 99.5) + 1e-9), 0, 1)
    thr = filters.threshold_local(sm, block_size=int(2 * p["L"]) | 1)
    raw = sm > thr
    pk = feature.peak_local_max(sm, min_distance=p["min_dist"], labels=measure.label(raw))
    markers = np.zeros(pre.shape, np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(filters.gaussian(edge, 1), markers)
    return {"edge": edge, "raw": raw, "markers": markers, "labels": lab}


# ------------------------------------------------------------- 4. ансамбль
def ensemble(pre, methods, p, weights=None):
    """Маркер = область, которую ВСЕ методы считают внутренностью одного зерна.

    Пересечение разбиений режет зерно там, где хоть один метод видит границу;
    эрозия убирает узкие спорные полосы. Остаются согласованные «ядра» зёрен,
    по ним — водораздел на взвешенной общей карте границ. Лишние разрезы потом
    убирает слияние по слабой границе (п. 5), т.е. итоговое решение о границе
    принимается по её реальной силе.
    """
    weights = weights or [1 / len(methods)] * len(methods)
    edge = sum(w * m["edge"] for w, m in zip(weights, methods))
    tol = morphology.disk(max(2, int(p["L"] / 15)))  # допуск ~ ширина границы
    bounds = [segmentation.find_boundaries(m["labels"], mode="thick") for m in methods]
    pair = np.zeros(pre.shape, np.int64)
    for m in methods:
        pair = pair * (int(m["labels"].max()) + 1) + m["labels"]
    any_b = np.logical_or.reduce(bounds)
    pair[any_b] = 0
    r = max(1, int(p["L"] / 10))
    core = ndi.binary_erosion(pair > 0, morphology.disk(r))
    pair[~core] = 0
    cores = measure.label(pair, connectivity=1, background=0)
    cores = remove_small(cores, max(4, p["min_area"] // 4))
    # согласованность границ (средняя по парам методов, с допуском)
    ag = []
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            bi, bj = bounds[i], bounds[j]
            ag.append(0.5 * ((bi & ndi.binary_dilation(bj, tol)).sum() / max(bi.sum(), 1) +
                             (bj & ndi.binary_dilation(bi, tol)).sum() / max(bj.sum(), 1)))
    # граница «подтверждена большинством» -> эталон силы для слияния
    votes = sum(ndi.binary_dilation(b, tol).astype(np.int8) for b in bounds)
    consensus = bounds[0] & (votes >= max(2, (len(bounds) + 1) // 2 + (len(bounds) > 2)))
    lab = segmentation.watershed(filters.gaussian(edge, 1), cores)
    return {"edge": edge, "raw": core, "markers": cores, "labels": lab,
            "boundary_agreement": float(np.mean(ag)), "consensus": consensus}


# --------------------------------------------------------- 5. постобработка
def detect_voids(img_flat, p):
    """Поры и трещины: заметно темнее поверхности И шире обычной границы."""
    med = np.median(img_flat)
    mad = np.median(np.abs(img_flat - med)) + 1e-6
    dark = img_flat < med - 5 * mad
    w = max(1, int(p["L"] / 15))           # обычная граница уже этого
    dark = ndi.binary_opening(dark, morphology.disk(w))
    return remove_small(dark, p["min_area"] // 2)


def boundary_strength(labels, edge):
    """Средняя сила границы между каждой парой соседей (RAG)."""
    rag = graph.rag_boundary(labels, edge.astype(float))
    return rag


def _merge_weak(labels, edge, thr):
    rag = graph.rag_boundary(labels, edge.astype(float))
    def _w(g, src, dst, n):
        # длинно-взвешенное среднее силы границы при слиянии регионов
        d0 = {"weight": 0.0, "count": 0}
        a, b = g[src].get(n, d0), g[dst].get(n, d0)
        c = a["count"] + b["count"]
        return {"count": c, "weight": (a["count"] * a["weight"] + b["count"] * b["weight"]) / max(c, 1)}
    def _m(g, src, dst):
        pass
    return graph.merge_hierarchical(labels, rag, thresh=thr, rag_copy=False,
                                    in_place_merge=True, merge_func=_m,
                                    weight_func=_w)


def postprocess(labels, edge, flat, p, merge_k=0.45, ref_mask=None):
    lab = labels.copy()
    # 5.1 поры/трещины — отдельный класс, не зёрна
    voids = detect_voids(flat, p)
    # 5.2 слияние ложно разделённых: граница слабее merge_k * типичной
    # эталон силы «настоящей» границы: границы, подтверждённые обоими методами.
    # Так порог не «уплывает» вниз, если исходное разбиение пересегментировано.
    if ref_mask is not None and ref_mask.sum() > 50:
        thr = merge_k * float(np.median(edge[ref_mask]))
    else:
        rag = graph.rag_boundary(lab, edge.astype(float))
        ws = np.array([d["weight"] for _, _, d in rag.edges(data=True)])
        thr = merge_k * np.median(ws) if len(ws) else 0
    lab = _merge_weak(lab, edge, thr) + 1
    # 5.3 мелкие фрагменты -> к соседу с самой слабой общей границей
    for _ in range(3):
        props = measure.regionprops(lab)
        small = [r.label for r in props if r.area < p["min_area"]]
        if not small:
            break
        rag = graph.rag_boundary(lab, edge.astype(float))
        for s in small:
            if s in rag and len(rag[s]):
                nb = min(rag[s], key=lambda n: rag[s][n]["weight"])
                lab[lab == s] = nb
    lab, _, _ = segmentation.relabel_sequential(lab)
    # 5.4 убираем поры из зёрен
    lab[voids] = 0
    # 5.5 краевые зёрна: сохраняем, но помечаем
    edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return lab, voids, edge_ids


# ------------------------------------------------------ 6. характеристики
def feret_min(coords):
    """Минимальный диаметр Фере (минимальная ширина выпуклой оболочки)."""
    if len(coords) < 3:
        return 0.0
    hull = cv2.convexHull(coords[:, ::-1].astype(np.float32))
    (_, _), (w, h), _ = cv2.minAreaRect(hull)
    return float(min(w, h))


def measure_grains(lab, edge, pre, edge_ids, px_nm):
    rows = []
    bnd = segmentation.find_boundaries(lab, mode="inner")
    for r in measure.regionprops(lab, intensity_image=pre):
        sl = r.slice
        own_b = bnd[sl] & r.image
        be = edge[sl][own_b]
        inner = edge[sl][r.image & ~own_b]
        k = px_nm or np.nan
        rows.append({
            "id": r.label,
            "cx_px": round(r.centroid[1], 1), "cy_px": round(r.centroid[0], 1),
            "area_px": int(r.area),
            "area_nm2": r.area * k ** 2,
            "perimeter_nm": r.perimeter * k,
            "eq_diameter_nm": r.equivalent_diameter_area * k,
            "major_axis_nm": r.axis_major_length * k,
            "minor_axis_nm": r.axis_minor_length * k,
            "eccentricity": r.eccentricity,
            "feret_max_nm": r.feret_diameter_max * k,
            "feret_min_nm": feret_min(r.coords) * k,
            "solidity": r.solidity,
            "circularity": 4 * np.pi * r.area / max(r.perimeter, 1) ** 2,
            "orientation_deg": np.degrees(r.orientation),
            "boundary_contrast": float(be.mean() - inner.mean()) if len(be) and len(inner) else np.nan,
            "boundary_uniformity_cv": float(be.std() / (be.mean() + 1e-9)) if len(be) else np.nan,
            "touches_border": r.label in edge_ids,
        })
    return rows


# --------------------------------------------------------------- всё вместе
# Подобрано по leave-one-out на ручной разметке (results/cv_summary.csv):
# A лучше всего при 0.45, ансамбль и RF — при 0.6.
DEFAULT_MERGE_K = {"A": 0.45, "B": 0.45, "C": 0.6, "E": 0.6}
# Веса карт границ в ансамбле: A — самый точный метод по CV, C — обучаемый,
# B — самый слабый, но с другим принципом (добавляет независимый голос).
ENSEMBLE_WEIGHTS = {2: [0.6, 0.4], 3: [0.5, 0.2, 0.3]}


def run(img, px_nm=None, merge_k=None, rf_model=None):
    """merge_k: None (подобранные значения), число (для всех методов) или dict."""
    if merge_k is None:
        merge_k = DEFAULT_MERGE_K
    if not isinstance(merge_k, dict):
        merge_k = {k: merge_k for k in "ABCE"}
    L = estimate_scale(img)
    p = params_from_scale(L)
    pre = preprocess(img, p)
    flat = filters.gaussian(img / np.maximum(filters.gaussian(img, p["bg_sigma"]), 1e-3), 1)
    out = {"params": p, "pre": pre,
           "A": method_ridge(pre, p), "B": method_intensity(pre, p)}
    keys = ["A", "B"]
    if rf_model is not None:
        import ml_boundary
        out["C"] = ml_boundary.method_rf(pre, p, rf_model)
        keys.append("C")
    w = ENSEMBLE_WEIGHTS[len(keys)]
    out["E"] = ensemble(pre, [out[k] for k in keys], p, w)
    consensus = out["E"]["consensus"]
    out["methods"] = keys + ["E"]
    for key in out["methods"]:
        lab, voids, edge_ids = postprocess(out[key]["labels"], out[key]["edge"], flat, p,
                                           merge_k[key], ref_mask=consensus)
        out[key].update(post=lab, voids=voids, edge_ids=edge_ids,
                        grains=measure_grains(lab, out[key]["edge"], pre, edge_ids, px_nm))
    return out
