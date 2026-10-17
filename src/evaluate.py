"""Оценка качества сегментации по ручной разметке.

Метрики:
  pixel  — Dice / IoU маски «зерно vs граница/фон»;
  object — сопоставление зёрен по IoU >= 0.5: precision, recall, F1,
           средний IoU найденных пар (аналог SQ из Panoptic Quality), PQ;
  split/merge — сколько эталонных зёрен раздроблено / слито
           (по перекрытию > 20% площади), т.е. пере- и недосегментация;
  boundary — Boundary F1 с допуском tol пикселей;
  size   — относительная ошибка медианного эквивалентного диаметра.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import measure, segmentation, morphology


# ------------------------------------------------------- чтение разметки
def load_gt(path, work_shape=None):
    """Маска в любом из форматов -> карта экземпляров (0 = граница/фон/пора).

    * RGB/палитра с множеством цветов: каждый цвет = зерно, чёрный/белый = фон;
    * бинарная (в т.ч. со сглаженными краями, в RGB, с альфа-каналом):
      меньший класс считается границей, зёрна = связные компоненты большего;
    * градации серого / 16 бит с многими значениями: значения = номера зёрен.
    """
    im = Image.open(path)
    mode = im.mode
    if mode in ("LA", "La", "PA"):
        L = np.asarray(im.convert("L"))
        A = np.asarray(im.getchannel("A"))
        # если яркость почти постоянна, а меняется прозрачность — маска в альфе
        a = A if (np.ptp(L) < 10 and np.ptp(A) > 10) else L
        return _from_gray(a, work_shape)
    if mode in ("RGBA",):
        rgb = np.asarray(im.convert("RGB")); A = np.asarray(im.getchannel("A"))
        if np.ptp(rgb) < 10 and np.ptp(A) > 10:
            return _from_gray(A, work_shape)
        im = im.convert("RGB")
    elif mode in ("P", "CMYK", "YCbCr"):
        im = im.convert("RGB")
    elif mode == "1":
        im = im.convert("L")
    a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., :3]
        if np.all(a[..., 0] == a[..., 1]) and np.all(a[..., 1] == a[..., 2]):
            return _from_gray(a[..., 0], work_shape)
        codes = a[..., 0].astype(np.int64) * 65536 + a[..., 1].astype(np.int64) * 256 + a[..., 2]
        u, inv = np.unique(codes, return_inverse=True)
        inv = inv.reshape(codes.shape)
        if len(u) <= 2 or _is_binary_like(a):
            return _from_gray(a.mean(2), work_shape)
        lab = inv + 1
        for i, c in enumerate(u):
            if c in (0, 0xFFFFFF):
                lab[inv == i] = 0
        return _crop(_split_disconnected(lab), work_shape)
    return _from_gray(a, work_shape)


def _is_binary_like(a):
    """Бинарная маска, возможно со сглаженными краями: почти все пиксели лежат
    у двух крайних значений, промежуточных (25–75% диапазона) меньше 10%.
    Для RGB проверяется яркость."""
    a = np.asarray(a, float)
    if a.ndim == 3:
        a = a.mean(2)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-9:
        return True
    t = (a - lo) / (hi - lo)
    if len(np.unique(a)) > 2 and ((t > 0.25) & (t < 0.75)).mean() >= 0.10:
        return False
    # и у краёв должно быть сосредоточено большинство пикселей
    return ((t <= 0.25) | (t >= 0.75)).mean() >= 0.9 and _few_levels_or_smooth(a)


def _few_levels_or_smooth(a):
    """Карта номеров зёрен тоже может иметь мало «средних» значений, если зёрен
    мало; отличаем её по числу уровней: у номеров уровни идут подряд и их много,
    у сглаженной бинарной маски почти все пиксели ровно на двух уровнях."""
    _, cnt = np.unique(a, return_counts=True)
    return np.sort(cnt)[-2:].sum() / a.size >= 0.8


def _from_gray(a, work_shape):
    a = np.asarray(a)
    vals = np.unique(a)
    if len(vals) <= 2 or _is_binary_like(a):
        thr = (float(a.min()) + float(a.max())) / 2
        fg = a > thr
        # зёрна = тот класс, который распадается на много отдельных кусков;
        # границы/фон образуют одну связную сеть
        n_fg = measure.label(fg, connectivity=1).max()
        n_bg = measure.label(~fg, connectivity=1).max()
        if n_bg > 3 * n_fg:
            fg = ~fg
        lab = measure.label(fg, connectivity=1)
    else:
        lab = _split_disconnected(a.astype(np.int64))
    return _crop(lab, work_shape)


def _split_disconnected(lab):
    """Один номер/цвет на два несвязных куска -> два зерна."""
    out = np.zeros_like(lab)
    k = 0
    for r in measure.regionprops(lab):
        cc = measure.label(r.image, connectivity=1)
        for j in range(1, cc.max() + 1):
            k += 1
            sl = r.slice
            out[sl][cc == j] = k
    return out


def _crop(lab, work_shape):
    """Маска больше рабочей области -> обрезаем (на маске была плашка).
    Маска меньше -> это верхняя часть снимка (проверено совмещением с
    изображением), остальное считается неразмеченным: дополняем -1."""
    if work_shape is None:
        return lab
    H, W = work_shape
    lab = lab[:H, :W]
    if lab.shape != (H, W):
        out = np.full((H, W), -1, np.int64)
        out[: lab.shape[0], : lab.shape[1]] = lab
        lab = out
    return lab


def annotated_region(lab):
    """Область, которую действительно размечали.

    Разметка частичная: зёрна обведены в некоторой зоне, вокруг — чёрное поле,
    которое НЕ является границей. Зона = зёрна, слитые через узкие межзёренные
    промежутки (замыкание) + заполнение внутренних дыр. Тёмные пиксели внутри
    зоны — границы, снаружи — «не размечено» и в метриках/обучении не участвуют.
    """
    fg = lab > 0
    r = max(3, int(round(4 * lab.shape[1] / 401)))
    pad = r + 1
    f = np.pad(fg, pad)
    region = ndi.binary_closing(f, morphology.disk(r))[pad:-pad, pad:-pad]
    region = ndi.binary_fill_holes(region)
    return region & (lab >= 0)


# ------------------------------------------------------------- метрики
def _iou_matrix(gt, pr):
    g = gt.ravel(); p = pr.ravel()
    m = (g > 0) & (p > 0)
    pairs, cnt = np.unique(np.stack([g[m], p[m]]), axis=1, return_counts=True)
    ag = np.bincount(g, minlength=gt.max() + 1)
    ap = np.bincount(p, minlength=pr.max() + 1)
    inter = cnt.astype(float)
    union = ag[pairs[0]] + ap[pairs[1]] - inter
    return pairs, inter, inter / union, ag, ap


def evaluate(gt, pr, tol=2, iou_thr=0.5, exclude_border=True, valid=None):
    gt = gt.astype(np.int64); pr = pr.astype(np.int64)
    gt = np.where(gt < 0, 0, gt)
    if valid is not None:
        # оставляем только предсказанные зёрна, лежащие в размеченной зоне
        # хотя бы на 50% своей площади; остальное снаружи не оцениваем
        cnt_in = np.bincount(pr[valid].ravel(), minlength=pr.max() + 1)
        cnt_all = np.bincount(pr.ravel(), minlength=pr.max() + 1)
        keep = np.flatnonzero(cnt_in >= 0.5 * np.maximum(cnt_all, 1))
        pr = np.where(np.isin(pr, keep[keep > 0]), pr, 0)
    if exclude_border:   # краевые зёрна не участвуют в количественном анализе
        gt = segmentation.clear_border(gt)
        keep = np.unique(pr[gt > 0])
        pr = np.where(np.isin(pr, keep[keep > 0]), pr, 0)
        pr = segmentation.clear_border(pr)
    gt, _, _ = segmentation.relabel_sequential(gt)
    pr, _, _ = segmentation.relabel_sequential(pr)
    res = {}
    # pixel
    G, P = gt > 0, pr > 0
    if valid is not None:
        G, P = G & valid, P & valid
    inter = (G & P).sum()
    res["dice"] = 2 * inter / max(G.sum() + P.sum(), 1)
    res["iou_px"] = inter / max((G | P).sum(), 1)
    # object
    pairs, inter_c, iou, ag, ap = _iou_matrix(gt, pr)
    ng, npr = gt.max(), pr.max()
    good = iou >= iou_thr                      # при IoU>0.5 пара единственна
    tp = int(good.sum())
    res.update(n_gt=int(ng), n_pred=int(npr), tp=tp,
               precision=tp / max(npr, 1), recall=tp / max(ng, 1))
    res["f1"] = 2 * tp / max(ng + npr, 1)
    res["sq"] = float(iou[good].mean()) if tp else 0.0
    res["pq"] = res["sq"] * res["f1"]
    # split / merge
    frac_g = inter_c / ag[pairs[0]]
    frac_p = inter_c / ap[pairs[1]]
    sig = (frac_g > 0.2) & (frac_p > 0.5)     # кусок предсказания внутри эталона
    split = np.bincount(pairs[0][sig], minlength=ng + 1)
    res["split_gt"] = int((split >= 2).sum())
    sig2 = (frac_p > 0.2) & (frac_g > 0.5)
    merge = np.bincount(pairs[1][sig2], minlength=npr + 1)
    res["merged_pred"] = int((merge >= 2).sum())
    # boundary F1 с допуском
    # Boundary F1 для масок с промежутком между зёрнами (ширина 3–10 пикс.):
    #  precision: пиксель найденной границы верен, если до промежутка <= tol;
    #  recall: точка осевой линии промежутка найдена, если до найденной
    #          границы <= tol + местная полуширина промежутка.
    gap = gt == 0
    if valid is not None:
        gap &= valid
    bp = segmentation.find_boundaries(pr, mode="inner")
    if gap.sum() > 0.02 * gap.size:
        bg = morphology.skeletonize(gap)
        halfw = ndi.distance_transform_edt(gap)
        if valid is not None:
            core = ndi.binary_erosion(valid, morphology.disk(int(np.percentile(halfw[bg], 90)) + tol + 2))
            bg, bp = bg & core, bp & core
        d_to_gap = ndi.distance_transform_edt(~gap)
        d_to_pred = ndi.distance_transform_edt(~bp)
        prec_b = (d_to_gap[bp] <= tol).mean() if bp.any() else 0.0
        rec_b = (d_to_pred[bg] <= tol + halfw[bg]).mean() if bg.any() else 0.0
    else:                       # маска-карта номеров без промежутков
        bg = segmentation.find_boundaries(gt)
        bp = segmentation.find_boundaries(pr)
        se = morphology.disk(tol)
        prec_b = (bp & ndi.binary_dilation(bg, se)).sum() / max(bp.sum(), 1)
        rec_b = (bg & ndi.binary_dilation(bp, se)).sum() / max(bg.sum(), 1)
    res["boundary_f1"] = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-9)
    # размер
    dg = np.median([r.equivalent_diameter_area for r in measure.regionprops(gt)]) if ng else np.nan
    dp = np.median([r.equivalent_diameter_area for r in measure.regionprops(pr)]) if npr else np.nan
    res["d_median_rel_err"] = (dp - dg) / dg
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in res.items()}
