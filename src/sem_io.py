"""Загрузка SEM-снимков, отделение служебной плашки, определение масштаба.

Поддерживает PNG/JPG/TIFF, RGB/палитру/градации серого, 8 и 16 бит.
"""
import csv
from pathlib import Path
import numpy as np
from PIL import Image

# Для Zeiss SUPRA: pixel_size[м] * Mag = const при ширине кадра 1024 px
# (проверено по метаданным Ba2_, Co2_, Cu_: 2.9492e-4 м)
ZEISS_K = 2.9492e-4
ZEISS_REF_WIDTH = 1024
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
# Служебная плашка занимает ~10% высоты кадра (проверено на TIFF 1024x768 и PNG 401x301).
# Тёмная область толще этого предела — скорее часть самого снимка, а не плашка.
MAX_PLATE_FRAC = 0.18


def parse_magnifications(text):
    """Текст magnifications.csv -> {имя снимка без расширения: увеличение, K X}.

    Терпимо к тому, как Excel сохраняет CSV: BOM в начале файла, разделитель
    ';' (русская локаль) или табуляция, десятичная запятая («53,04»), пробелы
    в заголовках, имя снимка с расширением. Некорректные строки пропускаются.
    """
    text = text.lstrip("\ufeff")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    delim = None
    for d in (",", ";", "\t"):
        head = [h.strip().lower() for h in lines[0].split(d)]
        if "image" in head and "mag_kx" in head:
            delim = d
            break
    if delim is None:
        return {}
    mags = {}
    for row in csv.DictReader(lines, delimiter=delim):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        name = r.get("image", "")
        if Path(name).suffix.lower() in IMG_EXT:
            name = Path(name).stem
        try:
            mags[name] = float(r["mag_kx"].replace(",", "."))
        except (KeyError, ValueError):
            continue
    return mags


def read_magnifications(path):
    """Читает magnifications.csv (колонки image, mag_kx). Пустой словарь = файл не распознан."""
    return parse_magnifications(Path(path).read_text(encoding="utf-8-sig", errors="replace"))


def load_gray(path):
    """Читает изображение любого формата и возвращает float32 в [0, 1]."""
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.asarray(im).astype(np.float32)
        return (a - a.min()) / max(a.max() - a.min(), 1)
    a = np.asarray(im.convert("L")).astype(np.float32)
    return a / 255.0


def find_work_area(img, dark_thr=15 / 255, ref_tol=0.1, max_plate_frac=MAX_PLATE_FRAC):
    """Возвращает номер первой строки служебной области (рабочая область = img[:cut]).

    1) плашка: первая строка в нижних 30% кадра, где >50% пикселей почти чёрные;
    2) над плашкой есть белая разделительная линия (в уменьшенных копиях она
       размыта), поэтому поднимаемся, пока средняя яркость строки резко
       отличается от типичной для поверхности.
    Если плашки нет, возвращает высоту изображения. Если найденная «плашка» толще
    max_plate_frac высоты кадра, это, скорее всего, тёмная область самого снимка:
    обрезка не делается (строку можно задать вручную в load_sem(cut_row=...)).
    """
    H = img.shape[0]
    dark = (img < dark_thr).mean(axis=1)
    cand = np.where((dark > 0.5) & (np.arange(H) > 0.7 * H))[0]
    if len(cand) == 0:
        return H
    cut = int(cand.min())
    row_mean = img.mean(axis=1)
    ref = np.median(row_mean[: int(0.6 * H)])
    while cut > 0 and abs(row_mean[cut - 1] - ref) > ref_tol:
        cut -= 1
    if H - cut > max_plate_frac * H:
        return H
    return cut


def pixel_size_from_tiff(path):
    """Размер пикселя (нм) из тега Zeiss CZ_SEM (34118) или None."""
    try:
        tags = Image.open(path).tag_v2
    except AttributeError:
        return None
    raw = tags.get(34118)
    if raw is None:
        return None
    raw = raw if isinstance(raw, str) else str(raw)
    # Подробный формат: "Image Pixel Size = 1.945 nm"
    for line in raw.splitlines():
        if "Pixel Size" in line and "=" in line:
            val, unit = line.split("=")[1].split()[:2]
            scale = {"pm": 1e-3, "nm": 1, "µm": 1e3, "um": 1e3}.get(unit, 1)
            return float(val) * scale
    # Компактный формат: 4-я строка — размер пикселя в метрах
    parts = [p for p in raw.replace("\r", "").split("\n") if p.strip()]
    try:
        v = float(parts[3])
        if 1e-12 < v < 1e-5:
            return v * 1e9
    except (IndexError, ValueError):
        pass
    return None


def pixel_size_from_mag(mag_kx, width_px):
    """Размер пикселя (нм) по увеличению (в K X) с учётом уменьшения копии."""
    return ZEISS_K / (mag_kx * 1e3) * 1e9 * ZEISS_REF_WIDTH / width_px


def load_sem(path, mag_kx=None, pixel_nm=None, cut_row=None):
    """Полная загрузка: рабочая область + масштаб.

    Приоритет масштаба: явно заданный pixel_nm -> метаданные TIFF -> увеличение.
    cut_row — ручная первая строка плашки (None = определить автоматически).
    Кроме данных возвращает "cut_source" (auto / manual / none) и список
    "warnings" — что пользователю стоит проверить.
    """
    path = Path(path)
    img = load_gray(path)
    H = img.shape[0]
    warnings = []
    if cut_row is not None:
        cut, cut_source = int(min(max(cut_row, 8), H)), "manual"
    else:
        cut = find_work_area(img)
        cut_source = "none" if cut == H else "auto"
        if cut == H and (img[int(0.82 * H):] < 15 / 255).mean() > 0.5:
            warnings.append("Внизу снимка большая тёмная область, не похожая на служебную "
                            "плашку, поэтому обрезка не выполнена. Если это плашка — "
                            "задайте первую строку плашки вручную.")
    source = "manual"
    if pixel_nm is None:
        pixel_nm = pixel_size_from_tiff(path)
        source = "tiff_metadata"
        # метаданные относятся к исходной ширине 1024 px
        if pixel_nm is not None and img.shape[1] != ZEISS_REF_WIDTH:
            pixel_nm *= ZEISS_REF_WIDTH / img.shape[1]
    if pixel_nm is None and mag_kx is not None:
        pixel_nm = pixel_size_from_mag(mag_kx, img.shape[1])
        source = "magnification"
    if pixel_nm is None:
        source = "unknown"
        warnings.append("Масштаб не определён (нет метаданных TIFF и увеличения): размеры "
                        "будут в пикселях. Укажите увеличение с плашки (Mag, K X), "
                        "нм/пиксель или magnifications.csv.")
    return {
        "name": path.stem,
        "image": img[:cut],
        "full": img,
        "cut_row": cut,
        "pixel_nm": pixel_nm,
        "scale_source": source,
        "cut_source": cut_source,
        "warnings": warnings,
    }