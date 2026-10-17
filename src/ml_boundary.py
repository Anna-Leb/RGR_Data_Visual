"""Метод C: обучаемый попиксельный классификатор границ (Random Forest).

Идея как в ilastik / Trainable Weka Segmentation, но с нормировкой масштаба:
каждый снимок масштабируется так, чтобы средний размер зерна стал L0 = 30 пикс.
Тогда один набор признаков и одна модель работают при любом увеличении.

Признаки (skimage.feature.multiscale_basic_features): яркость, края (Собель),
текстура (собственные числа Гессе) на масштабах σ = 1…8 пикселей.
Классы: 1 = внутренность зерна, 2 = межзёренная граница (и поры).
Выход: карта вероятности границы P(x) в [0, 1] — стадия «Probability map».
"""
import numpy as np
import joblib
from scipy import ndimage as ndi
from skimage import feature, transform, segmentation, morphology, measure, filters
from sklearn.ensemble import RandomForestClassifier

L0 = 30.0


def _rescale(img, L, order=1):
    k = L0 / L
    shape = (max(8, round(img.shape[0] * k)), max(8, round(img.shape[1] * k)))
    return transform.resize(img, shape, order=order, preserve_range=True,
                            anti_aliasing=(order > 0 and k < 1)), k


def features(img, L):
    small, _ = _rescale(img, L)
    f = feature.multiscale_basic_features(small, intensity=True, edges=True,
                                          texture=True, sigma_min=1, sigma_max=8,
                                          num_sigma=5)
    return f, small.shape


def targets(gt_lab, shape, L, valid=None):
    """Эталонная маска -> классы на канонической сетке (0 = не обучаем)."""
    lab = transform.resize(gt_lab, shape, order=0, preserve_range=True,
                           anti_aliasing=False).astype(np.int64)
    lab[lab < 0] = 0
    bnd = segmentation.find_boundaries(lab, mode="thick") | (lab == 0)
    bnd = ndi.binary_dilation(bnd, morphology.disk(1))
    inner = ndi.binary_erosion(~bnd, morphology.disk(2))
    y = np.zeros(shape, np.uint8)
    y[inner] = 1
    y[bnd] = 2
    if valid is not None:
        v = transform.resize(valid.astype(float), shape, order=0,
                             preserve_range=True, anti_aliasing=False) > 0.5
        y[~v] = 0
    return y


def train(samples, n_per_image=40000, seed=0):
    """samples: список (img_pre, gt_labels, L[, valid])."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for smp in samples:
        img, gt, L = smp[:3]
        valid = smp[3] if len(smp) > 3 else None
        f, shp = features(img, L)
        y = targets(gt, shp, L, valid)
        for c in (1, 2):            # сбалансированная выборка классов
            idx = np.flatnonzero(y.ravel() == c)
            idx = rng.choice(idx, min(len(idx), n_per_image // 2), replace=False)
            X.append(f.reshape(-1, f.shape[-1])[idx]); Y.append(np.full(len(idx), c))
    clf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5,
                                 n_jobs=-1, random_state=seed, class_weight="balanced")
    clf.fit(np.concatenate(X), np.concatenate(Y))
    return clf


def predict_proba(clf, img, L):
    f, shp = features(img, L)
    p = clf.predict_proba(f.reshape(-1, f.shape[-1]))[:, list(clf.classes_).index(2)]
    p = p.reshape(shp)
    return transform.resize(p, img.shape, order=1, preserve_range=True).astype(np.float32)


def method_rf(pre, p, clf):
    prob = predict_proba(clf, pre, p["L"])
    raw = prob < 0.5
    from pipeline import remove_small
    raw = remove_small(raw, p["min_area"] // 2)
    dist = ndi.distance_transform_edt(raw)
    pk = feature.peak_local_max(dist, min_distance=p["min_dist"],
                                threshold_abs=p["L"] / 10, labels=measure.label(raw))
    markers = np.zeros(pre.shape, np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(filters.gaussian(prob, 1), markers)
    return {"edge": prob, "raw": raw, "markers": markers, "labels": lab}


save = joblib.dump
load = joblib.load
