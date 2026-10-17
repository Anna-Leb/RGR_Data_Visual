"""Загрузка SEM-снимков, отделение служебной плашки, определение масштаба.

Поддерживает PNG/JPG/TIFF, RGB/палитру/градации серого, 8 и 16 бит.
"""
from pathlib import Path
import numpy as np
from PIL import Image

# Для Zeiss SUPRA: pixel_size[м] * Mag = const при ширине кадра 1024 px
# (проверено по метаданным Ba2_, Co2_, Cu_: 2.9492e-4 м)
ZEISS_K = 2.9492e-4
ZEISS_REF_WIDTH = 1024


def load_gray(path):
    """Читает изображение любого формата и возвращает float32 в [0, 1]."""
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.asarray(im).astype(np.float32)
        return (a - a.min()) / max(a.max() - a.min(), 1)
    a = np.asarray(im.convert("L")).astype(np.float32)
    return a / 255.0


def find_work_area(img, dark_thr=15 / 255, ref_tol=0.1):
    """Возвращает номер первой строки служебной области (рабочая область = img[:cut]).

    1) плашка: первая строка в нижних 30% кадра, где >50% пикселей почти чёрные;
    2) над плашкой есть белая разделительная линия (в уменьшенных копиях она
       размыта), поэтому поднимаемся, пока средняя яркость строки резко
       отличается от типичной для поверхности.
    Если плашки нет, возвращает высоту изображения.
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


def load_sem(path, mag_kx=None, pixel_nm=None):
    """Полная загрузка: рабочая область + масштаб.

    Приоритет масштаба: явно заданный pixel_nm -> метаданные TIFF -> увеличение.
    """
    path = Path(path)
    img = load_gray(path)
    cut = find_work_area(img)
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
    return {
        "name": path.stem,
        "image": img[:cut],
        "full": img,
        "cut_row": cut,
        "pixel_nm": pixel_nm,
        "scale_source": source,
    }
