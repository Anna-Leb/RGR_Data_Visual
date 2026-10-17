"""Многоклассовая разметка, ручная правка и экспорт результатов.

Правки хранятся как список операций с КООРДИНАТАМИ точек, а не с номерами
зёрен. Поэтому их можно сохранить в JSON и заново применить даже после
повторного расчёта с другими параметрами: номер зерна каждый раз ищется
по точке, в которую кликнул пользователь.

Операции:
  {"op": "delete", "p": [y, x]}                  — убрать объект (станет фоном)
  {"op": "merge",  "p": [y, x], "q": [y, x]}     — слить два соседних зерна
  {"op": "split",  "p": [y, x], "q": [y, x]}     — разделить зерно по двум центрам
  {"op": "class",  "p": [y, x], "cls": "particle"} — сменить класс объекта
"""
import io
import json
import zipfile
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage import segmentation, morphology, filters

# ------------------------------------------------------------------ классы
CLASSES = {
    0: ("фон / граница", (0, 0, 0)),
    1: ("зерно", (230, 60, 60)),
    2: ("краевое зерно", (250, 200, 30)),
    3: ("пора / трещина", (40, 120, 255)),
    4: ("частица на поверхности", (60, 200, 90)),
    5: ("исключено", (150, 150, 150)),
}
CLS_ID = {"grain": 1, "edge": 2, "void": 3, "particle": 4, "excluded": 5}
CLS_NAME = {v: k for k, v in CLS_ID.items()}
USER_CLASSES = ["grain", "particle", "void", "excluded"]   # что может выбрать пользователь


def _id_at(lab, p):
    y, x = int(p[0]), int(p[1])
    if not (0 <= y < lab.shape[0] and 0 <= x < lab.shape[1]):
        return 0
    i = int(lab[y, x])
    if i:
        return i
    # клик попал на линию границы — берём ближайшее зерно в радиусе 3 пикс.
    y0, y1, x0, x1 = max(0, y - 3), y + 4, max(0, x - 3), x + 4
    win = lab[y0:y1, x0:x1]
    if win.max() == 0:
        return 0
    yy, xx = np.nonzero(win)
    k = np.argmin((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2)
    return int(win[yy[k], xx[k]])


def _split(lab, edge, gid, p, q):
    mask = lab == gid
    mk = np.zeros(lab.shape, np.int32)
    for k, pt in enumerate((p, q), start=1):
        y, x = int(pt[0]), int(pt[1])
        rr = slice(max(0, y - 2), y + 3); cc = slice(max(0, x - 2), x + 3)
        mk[rr, cc] = k
    mk[~mask] = 0
    if len(np.unique(mk)) < 3:          # одна из точек вне зерна
        return lab, False
    part = segmentation.watershed(filters.gaussian(edge, 1), mk, mask=mask)
    new_id = int(lab.max()) + 1
    lab = lab.copy()
    lab[part == 2] = new_id
    return lab, True


def apply_ops(base_lab, edge, ops, base_cls=None):
    """Применяет правки к исходной разметке. Возвращает (labels, classes, log)."""
    lab = base_lab.astype(np.int32).copy()
    cls = dict(base_cls or {})
    log = []
    for op in ops:
        kind = op["op"]
        a = _id_at(lab, op["p"])
        if not a:
            log.append(f"{kind}: в точке {op['p']} нет объекта — пропущено"); continue
        if kind == "delete":
            lab[lab == a] = 0; cls.pop(a, None)
            log.append(f"удалён объект {a}")
        elif kind == "class":
            cls[a] = op["cls"]; log.append(f"объект {a} -> {op['cls']}")
        elif kind == "merge":
            b = _id_at(lab, op["q"])
            if not b or b == a:
                log.append("слияние: вторая точка не на другом зерне — пропущено"); continue
            lab[lab == b] = a
            # закрываем линию между ними: пиксели фона, граничащие с обоими
            lab[(lab == 0) & ndi.binary_closing(lab == a, morphology.disk(2))] = a
            cls.pop(b, None)
            log.append(f"слиты {a} и {b}")
        elif kind == "split":
            if _id_at(lab, op["q"]) != a:
                log.append("разделение: обе точки должны быть внутри одного зерна — пропущено"); continue
            lab, ok = _split(lab, edge, a, op["p"], op["q"])
            log.append(f"зерно {a} разделено" if ok else "разделение не удалось")
    return lab, cls, log


def class_map(lab, cls, voids, edge_ids):
    """Многоклассовая карта: 0 фон/граница, 1 зерно, 2 краевое, 3 пора, 4 частица, 5 исключено."""
    cm = np.zeros(lab.shape, np.uint8)
    ids = np.unique(lab); ids = ids[ids > 0]
    lut = np.zeros(int(lab.max()) + 1, np.uint8)
    for i in ids:
        c = cls.get(int(i))
        if c is None:
            c = "edge" if int(i) in edge_ids else "grain"
        lut[i] = CLS_ID[c]
    cm = lut[lab]
    cm[voids & (cm == 0)] = CLS_ID["void"]
    return cm


def edge_ids_of(lab):
    return set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}


def render(img, lab, cm, show_ids=False, scale=1, highlight=None):
    """Снимок + контуры, цвет контура = класс объекта. Возвращает uint8 RGB."""
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if scale != 1:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        lab = cv2.resize(lab.astype(np.int32), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cm = cv2.resize(cm, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    rgb = np.dstack([g] * 3).astype(np.float32)
    for c in (3, 4, 5):                                  # заливка не-зёрен
        m = cm == c
        rgb[m] = rgb[m] * 0.45 + np.array(CLASSES[c][1]) * 0.55
    b = segmentation.find_boundaries(lab, mode="inner")
    for c in (1, 2, 4, 5):
        m = b & (cm == c)
        rgb[m] = CLASSES[c][1]
    if highlight:
        m = np.isin(lab, list(highlight))
        rgb[m] = rgb[m] * 0.5 + np.array([255, 0, 255]) * 0.5
    rgb = rgb.astype(np.uint8)
    if show_ids:
        from skimage.measure import regionprops
        fs = 0.3 * max(1, scale) * (1 if img.shape[1] < 600 else 1.3)
        for r in regionprops(lab):
            y, x = r.centroid
            cv2.putText(rgb, str(r.label), (int(x) - 6, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def class_map_rgb(cm):
    out = np.zeros(cm.shape + (3,), np.uint8)
    for c, (_, col) in CLASSES.items():
        out[cm == c] = col
    return out


# ----------------------------------------------------------- сохранение
def corrections_json(image_name, method, ops, extra=None):
    d = {"image": image_name, "method": method, "ops": ops}
    if extra:
        d.update(extra)
    return json.dumps(d, ensure_ascii=False, indent=1)


def load_corrections(text):
    d = json.loads(text)
    return d.get("ops", []), d


def png_bytes(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); return buf.getvalue()


def csv_bytes(rows):
    import csv
    if not rows:
        return b""
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=list(rows[0].keys()), delimiter=";")
    w.writeheader(); w.writerows(rows)
    return s.getvalue().encode("utf-8-sig")


def zip_bytes(files):
    """files: {имя_в_архиве: bytes}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()
