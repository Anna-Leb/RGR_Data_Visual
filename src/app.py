"""Интерфейс программы сегментации зёрен на SEM-снимках.

Запуск из корня проекта:
    streamlit run src/app.py
Откроется вкладка браузера (обычно http://localhost:8501).
"""
import sys
import csv
import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import segmentation, color, measure

sys.path.insert(0, str(Path(__file__).parent))
import pipeline
import editing as ed
from sem_io import load_sem
from evaluate import evaluate, load_gt, annotated_region

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAVE_CLICK = True
except ImportError:
    HAVE_CLICK = False

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "rf_boundary.joblib"
METHOD_NAMES = {"A": "A — гребни (Sato) + водораздел",
                "B": "B — яркость зерна + водораздел",
                "C": "C — Random Forest (обучаемый)",
                "E": "E — ансамбль (итоговый)"}
IMG_TYPES = ["png", "tif", "tiff", "jpg", "jpeg", "bmp"]

st.set_page_config(page_title="Сегментация зёрен SEM", layout="wide")


# ======================================================= расчёт (кэшируется)
@st.cache_resource(show_spinner=False)
def get_model(path):
    import ml_boundary
    return ml_boundary.load(path)


def _to_tmp(name, data):
    suffix = Path(name).suffix or ".png"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data); f.close()
    return f.name


@st.cache_data(show_spinner=False, max_entries=64)
def load_image(name, data, mag_kx, pixel_nm):
    d = load_sem(_to_tmp(name, data), mag_kx=mag_kx or None, pixel_nm=pixel_nm or None)
    return {"image": d["image"], "full": d["full"], "cut_row": d["cut_row"],
            "pixel_nm": d["pixel_nm"], "scale_source": d["scale_source"]}


@st.cache_data(show_spinner=False, max_entries=64)
def process(name, data, mag_kx, pixel_nm, use_model, merge_items):
    d = load_image(name, data, mag_kx, pixel_nm)
    clf = get_model(str(MODEL_PATH)) if use_model and MODEL_PATH.exists() else None
    out = pipeline.run(d["image"], d["pixel_nm"], merge_k=dict(merge_items), rf_model=clf)
    res = {"params": out["params"], "pre": out["pre"], "methods": out["methods"]}
    for k in out["methods"]:
        M = out[k]
        res[k] = {key: M[key] for key in ("edge", "labels", "post", "voids", "edge_ids", "grains")}
    res["agreement"] = out["E"]["boundary_agreement"]
    return res


# ============================================================ боковая панель
st.sidebar.title("Сегментация зёрен")
files = st.sidebar.file_uploader("Перетащите снимки (PNG, TIFF, JPG; 8/16 бит)",
                                 type=IMG_TYPES, accept_multiple_files=True)
mag_file = st.sidebar.file_uploader("magnifications.csv (необязательно)", type=["csv"])
mags = {}
if mag_file is not None:
    for r in csv.DictReader(io.StringIO(mag_file.getvalue().decode("utf-8-sig"))):
        try:
            mags[r["image"]] = float(r["mag_kx"])
        except (KeyError, ValueError):
            pass

use_model = st.sidebar.checkbox("Использовать обучаемый метод C",
                                value=MODEL_PATH.exists(), disabled=not MODEL_PATH.exists(),
                                help="Нужна модель models/rf_boundary.joblib (создаётся cross_validate.py)")
with st.sidebar.expander("Параметры постобработки"):
    st.caption("Порог слияния соседних зёрен по силе границы (доля от силы "
               "подтверждённых границ). Больше — меньше дробления, больше слипания.")
    mk = {k: st.slider(f"merge_k, метод {k}", 0.1, 1.0, float(v), 0.05)
          for k, v in pipeline.DEFAULT_MERGE_K.items()}
merge_items = tuple(sorted(mk.items()))

if not files:
    st.title("Сегментация зёрен на SEM-снимках")
    st.info("Перетащите один или несколько снимков в поле слева.\n\n"
            "Программа сама отрежет служебную плашку, определит масштаб "
            "(у TIFF — из метаданных, у PNG — по увеличению из magnifications.csv "
            "или введённому вручную) и выделит зёрна четырьмя методами.")
    st.stop()

names = [f.name for f in files]
by_name = {f.name: f for f in files}
cur = st.sidebar.selectbox("Текущий снимок", names)
f = by_name[cur]
stem = Path(cur).stem

st.sidebar.markdown("**Масштаб**")
scale_mode = st.sidebar.radio("Источник масштаба",
                              ["авто (TIFF / таблица)", "увеличение, K X", "нм на пиксель"],
                              key=f"sm_{cur}")
mag_kx, pixel_nm = mags.get(stem), None
if scale_mode == "увеличение, K X":
    mag_kx = st.sidebar.number_input("Mag, K X (с плашки снимка)", 1.0, 2000.0,
                                     float(mags.get(stem, 50.0)), key=f"mag_{cur}")
elif scale_mode == "нм на пиксель":
    pixel_nm = st.sidebar.number_input("нм / пиксель", 0.01, 1000.0, 5.0, key=f"px_{cur}")

gt_file = st.sidebar.file_uploader("Эталонная маска для этого снимка (необязательно)",
                                   type=IMG_TYPES, key=f"gt_{cur}")

with st.spinner(f"Обработка {cur}…"):
    info = load_image(cur, f.getvalue(), mag_kx, pixel_nm)
    R = process(cur, f.getvalue(), mag_kx, pixel_nm, use_model, merge_items)
img = info["image"]
px = info["pixel_nm"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Размер рабочей области", f"{img.shape[1]}×{img.shape[0]}")
c2.metric("Масштаб, нм/пикс.", f"{px:.3f}" if px else "не задан")
c3.metric("Размер зерна L, пикс.", f"{R['params']['L']:.1f}")
c4.metric("Согласие методов по границам", f"{R['agreement']:.0%}")
st.caption(f"Источник масштаба: {info['scale_source']}. "
           f"Служебная плашка отрезана со строки {info['cut_row']}.")

tab_stages, tab_edit, tab_grains, tab_quality, tab_batch = st.tabs(
    ["Стадии обработки", "Ручная правка", "Характеристики зёрен",
     "Качество по эталону", "Пакетная обработка"])


# ========================================================= 1. стадии обработки
with tab_stages:
    meth = st.radio("Метод", R["methods"], format_func=METHOD_NAMES.get, horizontal=True)
    M = R[meth]
    w = 420
    a, b, c = st.columns(3)
    a.image(img, caption="1. Original", width=w, clamp=True)
    b.image(R["pre"], caption="2. Preprocessed", width=w, clamp=True)
    e = M["edge"] / (M["edge"].max() + 1e-9)
    c.image((plt.cm.magma(e)[..., :3] * 255).astype(np.uint8), width=w,
            caption="3. Probability map (RF)" if meth == "C" else "3. Edge / boundary map")
    a, b, c = st.columns(3)
    a.image(segmentation.mark_boundaries(img, M["labels"], color=(1, 0, 0)), width=w, clamp=True,
            caption=f"4. Raw segmentation ({int(M['labels'].max())} объектов)")
    b.image(color.label2rgb(M["post"], bg_label=0, alpha=1), width=w, clamp=True,
            caption=f"5. Postprocessed ({int(M['post'].max())} объектов)")
    cm0 = ed.class_map(M["post"], {}, M["voids"], M["edge_ids"])
    c.image(ed.render(img, M["post"], cm0), width=w, caption="6. Overlay on original")
    st.caption("Красный — зерно, жёлтый — краевое зерно (исключается из статистики), "
               "синий — пора/трещина.")


# ============================================================== 2. ручная правка
ss = st.session_state
key_ops = f"ops::{cur}"
ss.setdefault(key_ops, [])
ss.setdefault(f"pending::{cur}", None)
ss.setdefault(f"lastclick::{cur}", None)

with tab_edit:
    left, right = st.columns([3, 1])
    with right:
        base_meth = st.selectbox("Исходная разметка", R["methods"],
                                 index=R["methods"].index("E"), format_func=METHOD_NAMES.get)
        action = st.radio("Действие по клику", [
            "показать номер", "удалить объект", "слить два зерна (2 клика)",
            "разделить зерно (2 клика внутри)", "назначить класс"])
        new_cls = None
        if action == "назначить класс":
            new_cls = st.selectbox("Класс", ed.USER_CLASSES,
                                   format_func=lambda c: ed.CLASSES[ed.CLS_ID[c]][0])
        show_ids = st.checkbox("Номера зёрен на снимке", value=False)
        zoom = st.slider("Увеличение изображения", 1, 4, 2 if img.shape[1] < 600 else 1)
        st.markdown("**Правки**")
        b1, b2 = st.columns(2)
        if b1.button("Отменить", disabled=not ss[key_ops]):
            ss[key_ops].pop(); ss[f"pending::{cur}"] = None; st.rerun()
        if b2.button("Сбросить", disabled=not ss[key_ops]):
            ss[key_ops] = []; ss[f"pending::{cur}"] = None; st.rerun()
        up = st.file_uploader("Загрузить правки (JSON)", type=["json"], key=f"js_{cur}")
        if up is not None and st.button("Применить загруженные правки"):
            ops, meta = ed.load_corrections(up.getvalue().decode("utf-8"))
            ss[key_ops] = ops; st.rerun()

    M = R[base_meth]
    lab, cls, log = ed.apply_ops(M["post"], M["edge"], ss[key_ops])
    edge_ids = ed.edge_ids_of(lab)
    cm = ed.class_map(lab, cls, M["voids"], edge_ids)
    pend = ss[f"pending::{cur}"]
    hl = {ed._id_at(lab, pend)} if pend else None
    view = ed.render(img, lab, cm, show_ids=show_ids, scale=zoom, highlight=hl)

    with left:
        click = None
        if HAVE_CLICK:
            from PIL import Image
            st.caption("Кликните по зерну на снимке. Фиолетовым подсвечен первый выбранный объект.")
            v = streamlit_image_coordinates(Image.fromarray(view), key=f"clk_{cur}_{len(ss[key_ops])}")
            if v and v != ss[f"lastclick::{cur}"]:
                ss[f"lastclick::{cur}"] = v
                click = [int(v["y"] / zoom), int(v["x"] / zoom)]
        else:
            st.image(view)
            st.warning("Для правки кликом установите пакет: pip install streamlit-image-coordinates. "
                       "Пока можно вводить координаты вручную.")
            cx, cy, cb = st.columns(3)
            x = cx.number_input("x, пикс.", 0, img.shape[1] - 1, 0)
            y = cy.number_input("y, пикс.", 0, img.shape[0] - 1, 0)
            if cb.button("Применить к точке"):
                click = [int(y), int(x)]

    if click is not None:
        gid = ed._id_at(lab, click)
        if action == "показать номер":
            st.toast(f"Объект №{gid}" if gid else "Здесь граница или фон")
        elif action == "удалить объект":
            ss[key_ops].append({"op": "delete", "p": click}); st.rerun()
        elif action == "назначить класс":
            ss[key_ops].append({"op": "class", "p": click, "cls": new_cls}); st.rerun()
        else:
            op = "merge" if action.startswith("слить") else "split"
            if pend is None:
                ss[f"pending::{cur}"] = click; st.rerun()
            else:
                ss[key_ops].append({"op": op, "p": pend, "q": click})
                ss[f"pending::{cur}"] = None; st.rerun()

    with right:
        if pend:
            st.info("Выбрана первая точка. Кликните вторую.")
        st.caption(f"Правок: {len(ss[key_ops])}")
        for line in log[-8:]:
            st.text("• " + line)
        lg = st.columns(1)[0]
        lg.markdown(" ".join(
            f"<span style='color:rgb{col}'>■</span> {nm}&nbsp;&nbsp;"
            for c, (nm, col) in ed.CLASSES.items() if c), unsafe_allow_html=True)

    # итоговые данные с учётом правок — используются в остальных вкладках
    grains = pipeline.measure_grains(lab, M["edge"], R["pre"], edge_ids, px)
    for g in grains:
        g["class"] = cls.get(g["id"], "edge" if g["id"] in edge_ids else "grain")
    ss[f"final::{cur}"] = {"lab": lab, "cm": cm, "grains": grains, "method": base_meth}

    st.markdown("**Сохранение результата**")
    files_out = {
        f"{stem}_corrections.json": ed.corrections_json(
            cur, base_meth, ss[key_ops],
            {"merge_k": dict(merge_items), "pixel_nm": px}).encode("utf-8"),
        f"{stem}_labels.png": ed.png_bytes(lab.astype(np.uint16)),
        f"{stem}_classes.png": ed.png_bytes(cm),
        f"{stem}_classes_color.png": ed.png_bytes(ed.class_map_rgb(cm)),
        f"{stem}_overlay.png": ed.png_bytes(ed.render(img, lab, cm)),
        f"{stem}_grains.csv": ed.csv_bytes(grains),
    }
    d1, d2 = st.columns(2)
    d1.download_button("Скачать всё (ZIP)", ed.zip_bytes(files_out), f"{stem}_result.zip")
    d2.download_button("Только правки (JSON)", files_out[f"{stem}_corrections.json"],
                       f"{stem}_corrections.json")


# ======================================================= 3. характеристики зёрен
with tab_grains:
    fin = ss.get(f"final::{cur}")
    grains = fin["grains"]
    inner = [g for g in grains if g["class"] == "grain"]
    st.caption(f"Разметка: метод {fin['method']} с учётом ручных правок. В статистику "
               f"входят только целые зёрна класса «зерно» ({len(inner)} из {len(grains)} объектов).")
    unit = "нм" if px else "пикс."
    k = px or 1.0
    if inner:
        d = np.array([g["eq_diameter_nm"] if px else g["area_px"] ** 0.5 * 1.128 for g in inner])
        fmax = np.array([g["feret_max_nm"] if px else np.nan for g in inner])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Зёрен", len(inner))
        m2.metric(f"Медианный диаметр, {unit}", f"{np.median(d):.0f}")
        m3.metric(f"Средний ± СКО, {unit}", f"{d.mean():.0f} ± {d.std():.0f}")
        m4.metric("Ср. эксцентриситет", f"{np.mean([g['eccentricity'] for g in inner]):.2f}")
        h1, h2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(d, bins=25, color="#c0392b", alpha=0.85)
        ax.set_xlabel(f"Эквивалентный диаметр, {unit}"); ax.set_ylabel("Число зёрен")
        h1.pyplot(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist([g["circularity"] for g in inner], bins=25, color="#2c7fb8", alpha=0.85)
        ax.set_xlabel("Округлость 4πS/P²"); ax.set_ylabel("Число зёрен")
        h2.pyplot(fig); plt.close(fig)
    st.dataframe(grains, height=380)
    st.download_button("Скачать таблицу (CSV)", ed.csv_bytes(grains), f"{stem}_grains.csv")


# ========================================================= 4. качество по эталону
with tab_quality:
    if gt_file is None:
        st.info("Загрузите в боковой панели ручную маску для этого снимка — "
                "здесь появятся метрики всех методов и результата после правок.")
    else:
        gt = load_gt(_to_tmp(gt_file.name, gt_file.getvalue()), img.shape)
        valid = annotated_region(gt)
        rows = []
        for m in R["methods"]:
            r = evaluate(gt, R[m]["post"], valid=valid)
            rows.append({"метод": METHOD_NAMES[m], **r})
        r = evaluate(gt, ss[f"final::{cur}"]["lab"], valid=valid)
        rows.append({"метод": "после ручной правки", **r})
        st.dataframe(rows)
        st.caption("F1/PQ — по зёрнам (совпадение при IoU ≥ 0,5); boundary_f1 — по границам; "
                   "split_gt — раздробленные эталонные зёрна; merged_pred — слитые. "
                   f"Оценивается только размеченная зона ({valid.mean():.0%} снимка), "
                   "краевые зёрна исключены.")
        a, b = st.columns(2)
        rgb = segmentation.mark_boundaries(img, np.maximum(gt, 0), color=(0, 1, 0))
        rgb = segmentation.mark_boundaries(rgb, ss[f"final::{cur}"]["lab"], color=(1, 0, 0))
        rgb[~valid] *= 0.45
        a.image(rgb, caption="Зелёный — эталон, красный — результат", clamp=True)


# ========================================================== 5. пакетная обработка
with tab_batch:
    st.write(f"Загружено снимков: {len(files)}. Масштаб PNG берётся из magnifications.csv "
             "(если загружен), у TIFF — из метаданных.")
    bm = st.selectbox("Метод для сводки", ["E", "A", "B"] + (["C"] if use_model and MODEL_PATH.exists() else []),
                      format_func=METHOD_NAMES.get)
    if st.button("Обработать все снимки"):
        prog = st.progress(0.0)
        summary, zfiles = [], {}
        for i, fl in enumerate(files):
            st_ = Path(fl.name).stem
            inf = load_image(fl.name, fl.getvalue(), mags.get(st_), None)
            Rb = process(fl.name, fl.getvalue(), mags.get(st_), None, use_model, merge_items)
            # ручные правки, если они есть для этого снимка, тоже применяются
            Mb = Rb[bm]
            labb, clsb, _ = ed.apply_ops(Mb["post"], Mb["edge"], ss.get(f"ops::{fl.name}", []))
            eids = ed.edge_ids_of(labb)
            cmb = ed.class_map(labb, clsb, Mb["voids"], eids)
            g = pipeline.measure_grains(labb, Mb["edge"], Rb["pre"], eids, inf["pixel_nm"])
            for x in g:
                x["class"] = clsb.get(x["id"], "edge" if x["id"] in eids else "grain")
            inn = [x for x in g if x["class"] == "grain"]
            dd = [x["eq_diameter_nm"] for x in inn]
            summary.append({
                "снимок": st_, "нм/пикс": round(inf["pixel_nm"], 3) if inf["pixel_nm"] else "",
                "зёрен": len(inn), "пор": int(measure.label(cmb == 3).max()),
                "медиана d, нм": round(float(np.median(dd)), 1) if dd and inf["pixel_nm"] else "",
                "среднее d, нм": round(float(np.mean(dd)), 1) if dd and inf["pixel_nm"] else "",
                "СКО d, нм": round(float(np.std(dd)), 1) if dd and inf["pixel_nm"] else "",
                "правок": len(ss.get(f"ops::{fl.name}", [])),
            })
            zfiles[f"{st_}/grains.csv"] = ed.csv_bytes(g)
            zfiles[f"{st_}/labels.png"] = ed.png_bytes(labb.astype(np.uint16))
            zfiles[f"{st_}/classes_color.png"] = ed.png_bytes(ed.class_map_rgb(cmb))
            zfiles[f"{st_}/overlay.png"] = ed.png_bytes(ed.render(inf["image"], labb, cmb))
            prog.progress((i + 1) / len(files), text=fl.name)
        zfiles["summary.csv"] = ed.csv_bytes(summary)
        ss["batch_summary"], ss["batch_zip"] = summary, ed.zip_bytes(zfiles)
    if "batch_summary" in ss:
        st.dataframe(ss["batch_summary"])
        st.download_button("Скачать результаты всех снимков (ZIP)", ss["batch_zip"], "batch_results.zip")
