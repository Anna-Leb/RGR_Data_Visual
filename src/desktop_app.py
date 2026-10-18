"""Настольное приложение «Сегментация зёрен на SEM-снимках» (PySide6 / Qt).

Запуск из корня проекта:
    python src/desktop_app.py

Возможности:
  * загрузка снимков перетаскиванием в окно, через меню «Файл» или целой папкой;
  * PNG / TIFF / JPG / BMP, 8 и 16 бит; масштаб из метаданных TIFF,
    из magnifications.csv, по увеличению или вручную (нм/пикс.);
  * 6 стадий обработки для каждого метода (A, B, C, E);
  * ручная правка кликом: удалить, слить, разделить, сменить класс; отмена;
    сохранение и загрузка правок (JSON);
  * многоклассовая карта: зерно / краевое зерно / пора / частица / исключено;
  * таблица характеристик зёрен, гистограммы, экспорт CSV;
  * метрики качества по эталонной маске;
  * пакетная обработка всех загруженных снимков с сохранением в папку.
"""
import sys
import os
import traceback
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QRunnable, QThreadPool, QPointF
from PySide6.QtGui import QImage, QPixmap, QAction, QColor, QBrush, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QListWidgetItem, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QProgressBar, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QSplitter, QPlainTextEdit, QHeaderView,
    QDockWidget, QScrollArea, QSpinBox,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import app_core as core
import editing as ed
import pipeline
from evaluate import evaluate, load_gt, annotated_region


# ============================================================ фоновые задачи
class _Signals(QObject):
    done = Signal(object)
    error = Signal(str)
    progress = Signal(int, str)


class Task(QRunnable):
    """Выполняет функцию в фоновом потоке, чтобы окно не «зависало»."""

    def __init__(self, fn, *args, with_progress=False, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw
        self.signals = _Signals()
        if with_progress:
            self.kw["progress"] = self.signals.progress.emit

    def run(self):
        try:
            res = self.fn(*self.args, **self.kw)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.done.emit((id(self), res))


def to_pixmap(arr):
    a = np.ascontiguousarray(arr)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if a.ndim == 2:
        h, w = a.shape
        q = QImage(a.data, w, h, w, QImage.Format_Grayscale8)
    else:
        a = np.ascontiguousarray(a[..., :3])
        h, w, _ = a.shape
        q = QImage(a.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


# ===================================================== просмотр с кликами
class ImageView(QGraphicsView):
    """Изображение с масштабированием колесом мыши и кликами в координатах снимка."""
    clicked = Signal(int, int)      # y, x в пикселях снимка

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.FastTransformation)
        self.scene().addItem(self.item)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        self._fitted = False

    def set_image(self, arr, keep_view=True):
        self.item.setPixmap(to_pixmap(arr))
        self.scene().setSceneRect(self.item.boundingRect())
        if not (keep_view and self._fitted):
            self.fit()

    def fit(self, *_):
        self.fitInView(self.item, Qt.KeepAspectRatio)
        self._fitted = True

    def wheelEvent(self, e):
        k = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(k, k)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            p = self.mapToScene(e.position().toPoint())
            if self.item.boundingRect().contains(p):
                self.clicked.emit(int(p.y()), int(p.x()))
            return
        if e.button() in (Qt.RightButton, Qt.MiddleButton):   # перетаскивание
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if getattr(self, "_pan", None) is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if getattr(self, "_pan", None) is not None:
            self._pan = None
            self.unsetCursor()
            return
        super().mouseReleaseEvent(e)


class MplCanvas(FigureCanvas):
    def __init__(self, w=6, h=4):
        self.fig = Figure(figsize=(w, h))
        super().__init__(self.fig)


def fill_table(table, rows, max_rows=5000):
    table.clear()
    if not rows:
        table.setRowCount(0); table.setColumnCount(0); return
    cols = list(rows[0].keys())
    table.setColumnCount(len(cols)); table.setHorizontalHeaderLabels(cols)
    table.setRowCount(min(len(rows), max_rows))
    for i, r in enumerate(rows[:max_rows]):
        for j, c in enumerate(cols):
            v = r[c]
            if isinstance(v, (float, np.floating)):
                v = f"{v:.4g}"
            table.setItem(i, j, QTableWidgetItem(str(v)))
    table.resizeColumnsToContents()


# ================================================================ состояние
class ImageState:
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        self.stem = self.path.stem
        self.scale_mode = "auto"        # auto | mag | px
        self.mag = None
        self.px = None
        self.result = None              # результат core.compute
        self.result_key = None
        self.ops = []
        self.pending = None
        self.method = "E"
        self.gt_path = None
        self.cut_row = None             # None = плашка определяется автоматически


# ================================================================ окно
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сегментация зёрен на SEM-снимках")
        self.resize(1500, 900)
        self.setAcceptDrops(True)
        self.pool = QThreadPool.globalInstance()
        self.tasks = []                 # ссылки на задачи, чтобы их не удалил сборщик мусора
        self.images = {}                # имя -> ImageState
        self.mags = {}
        self.cur = None
        self._zoom = 1
        self._callbacks = {}
        self._build_menu()
        self._build_left()
        self._build_tabs()
        self.statusBar().showMessage("Перетащите снимки в окно или откройте их через меню «Файл».")

    # ------------------------------------------------------------ интерфейс
    def _build_menu(self):
        m = self.menuBar().addMenu("Файл")
        for text, slot, sc in [
            ("Открыть снимки…", self.open_files, "Ctrl+O"),
            ("Открыть папку…", self.open_folder, "Ctrl+Shift+O"),
            ("Загрузить magnifications.csv…", self.open_mags, None),
            ("Загрузить эталонную маску для снимка…", self.open_gt, None),
            (None, None, None),
            ("Сохранить результат снимка…", self.save_current, "Ctrl+S"),
            ("Выход", self.close, "Ctrl+Q"),
        ]:
            if text is None:
                m.addSeparator(); continue
            a = QAction(text, self); a.triggered.connect(slot)
            if sc:
                a.setShortcut(sc)
            m.addAction(a)
        h = self.menuBar().addMenu("Справка")
        a = QAction("Как пользоваться", self); a.triggered.connect(self.show_help); h.addAction(a)

    def _build_left(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>Снимки</b> (перетащите файлы в окно)"))
        self.list = QListWidget(); self.list.currentItemChanged.connect(self.on_select)
        lay.addWidget(self.list, 2)

        g = QGroupBox("Масштаб текущего снимка"); f = QFormLayout(g)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["авто (TIFF / таблица)", "по увеличению, K X", "вручную, нм/пикс."])
        self.scale_combo.currentIndexChanged.connect(self.on_scale_mode)
        self.scale_val = QDoubleSpinBox(); self.scale_val.setRange(0.001, 5000); self.scale_val.setDecimals(3)
        self.scale_val.setEnabled(False)
        self.scale_info = QLabel("—"); self.scale_info.setWordWrap(True)
        f.addRow(self.scale_combo); f.addRow("Значение:", self.scale_val); f.addRow(self.scale_info)
        lay.addWidget(g)

        g = QGroupBox("Служебная плашка"); f = QFormLayout(g)
        self.cut_check = QCheckBox("Задать первую строку плашки вручную")
        self.cut_check.setToolTip("Если автоопределение ошиблось (плашку не нашли или обрезали лишнее)")
        self.cut_spin = QSpinBox(); self.cut_spin.setRange(8, 20000); self.cut_spin.setEnabled(False)
        self.cut_check.toggled.connect(self.on_cut_toggle)
        self.cut_spin.valueChanged.connect(self.on_cut_value)
        f.addRow(self.cut_check); f.addRow("Первая строка плашки:", self.cut_spin)
        lay.addWidget(g)

        g = QGroupBox("Параметры"); f = QFormLayout(g)
        self.use_model = QCheckBox("Метод C (Random Forest)")
        self.use_model.setChecked(core.MODEL_PATH.exists())
        self.use_model.setEnabled(core.MODEL_PATH.exists())
        if not core.MODEL_PATH.exists():
            self.use_model.setToolTip("Нет models/rf_boundary.joblib — запустите cross_validate.py")
        f.addRow(self.use_model)
        self.mk = {}
        for k, v in pipeline.DEFAULT_MERGE_K.items():
            s = QDoubleSpinBox(); s.setRange(0.05, 1.5); s.setSingleStep(0.05); s.setValue(v)
            s.setToolTip("Порог слияния соседних зёрен по силе границы: больше — меньше дробления")
            self.mk[k] = s; f.addRow(f"merge_k {k}:", s)
        lay.addWidget(g)

        self.btn_run = QPushButton("Обработать снимок"); self.btn_run.clicked.connect(self.run_current)
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        lay.addWidget(self.btn_run)
        self.busy = QProgressBar(); self.busy.setRange(0, 0); self.busy.setVisible(False)
        lay.addWidget(self.busy)

        self.info = QLabel(""); self.info.setWordWrap(True)
        lay.addWidget(self.info)
        lay.addStretch(1)
        dock = QDockWidget("Управление", self); dock.setWidget(w)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _build_tabs(self):
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)

        # 1. стадии
        t = QWidget(); v = QVBoxLayout(t)
        row = QHBoxLayout(); row.addWidget(QLabel("Метод:"))
        self.stage_method = QComboBox(); self.stage_method.currentIndexChanged.connect(self.draw_stages)
        row.addWidget(self.stage_method); row.addStretch(1)
        b = QPushButton("Сохранить рисунок…"); b.clicked.connect(self.save_stages_png); row.addWidget(b)
        v.addLayout(row)
        self.stage_canvas = MplCanvas(13, 7.5); v.addWidget(self.stage_canvas)
        self.tabs.addTab(t, "1. Стадии обработки")

        # 2. ручная правка
        t = QWidget(); h = QHBoxLayout(t)
        self.view = ImageView(); self.view.clicked.connect(self.on_click)
        h.addWidget(self.view, 4)
        side = QWidget(); s = QVBoxLayout(side)
        f = QFormLayout()
        self.edit_method = QComboBox(); self.edit_method.currentIndexChanged.connect(self.on_edit_method)
        f.addRow("Исходная разметка:", self.edit_method); s.addLayout(f)
        gb = QGroupBox("Действие по клику"); gv = QVBoxLayout(gb)
        self.actions = QButtonGroup(self)
        for i, txt in enumerate(["показать номер и размер", "удалить объект",
                                 "слить два зерна (2 клика)", "разделить зерно (2 клика внутри)",
                                 "назначить класс"]):
            r = QRadioButton(txt); r.setChecked(i == 0); self.actions.addButton(r, i); gv.addWidget(r)
        self.cls_combo = QComboBox()
        for c in ed.USER_CLASSES:
            self.cls_combo.addItem(ed.CLASSES[ed.CLS_ID[c]][0], c)
        gv.addWidget(self.cls_combo)
        self.actions.idClicked.connect(lambda _: self.cancel_pending())
        s.addWidget(gb)
        self.show_ids = QCheckBox("Номера зёрен"); self.show_ids.toggled.connect(self.draw_edit)
        s.addWidget(self.show_ids)
        r = QHBoxLayout()
        for txt, fn in [("Отменить", self.undo), ("Сбросить", self.reset_ops), ("Вписать", self.view.fit)]:
            b = QPushButton(txt); b.clicked.connect(fn); r.addWidget(b)
        s.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("Сохранить правки…"); b.clicked.connect(self.save_ops); r.addWidget(b)
        b = QPushButton("Загрузить правки…"); b.clicked.connect(self.load_ops); r.addWidget(b)
        s.addLayout(r)
        legend = " ".join(f"<span style='color:rgb{col}'>■</span>&nbsp;{nm}<br>"
                          for c, (nm, col) in ed.CLASSES.items() if c)
        s.addWidget(QLabel(legend))
        self.edit_hint = QLabel(""); self.edit_hint.setWordWrap(True)
        self.edit_hint.setStyleSheet("color: #b05c00;")
        s.addWidget(self.edit_hint)
        s.addWidget(QLabel("Журнал правок:"))
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); s.addWidget(self.log, 1)
        s.addWidget(QLabel("<small>Колесо мыши — масштаб, правая кнопка — перетаскивание.</small>"))
        h.addWidget(side, 1)
        self.tabs.addTab(t, "2. Ручная правка")

        # 3. характеристики
        t = QWidget(); v = QVBoxLayout(t)
        self.stats_label = QLabel(""); self.stats_label.setStyleSheet("font-size: 13px;")
        v.addWidget(self.stats_label)
        sp = QSplitter(Qt.Vertical)
        self.hist_canvas = MplCanvas(12, 3.2); sp.addWidget(self.hist_canvas)
        self.grain_table = QTableWidget(); self.grain_table.setSortingEnabled(True)
        sp.addWidget(self.grain_table); v.addWidget(sp, 1)
        b = QPushButton("Сохранить таблицу (CSV)…"); b.clicked.connect(self.save_grains_csv)
        v.addWidget(b)
        self.tabs.addTab(t, "3. Характеристики зёрен")

        # 4. качество
        t = QWidget(); v = QVBoxLayout(t)
        r = QHBoxLayout()
        b = QPushButton("Загрузить эталонную маску…"); b.clicked.connect(self.open_gt); r.addWidget(b)
        self.gt_label = QLabel("Эталон не загружен"); r.addWidget(self.gt_label); r.addStretch(1)
        v.addLayout(r)
        self.q_table = QTableWidget(); v.addWidget(self.q_table, 1)
        self.q_view = ImageView(); v.addWidget(self.q_view, 2)
        v.addWidget(QLabel("<small>Зелёный — эталон, красный — результат (с правками). "
                           "Оценивается только размеченная зона, краевые зёрна исключены.</small>"))
        self.tabs.addTab(t, "4. Качество по эталону")

        # 5. пакетная обработка
        t = QWidget(); v = QVBoxLayout(t)
        r = QHBoxLayout()
        r.addWidget(QLabel("Метод для сводки:"))
        self.batch_method = QComboBox()
        for m in ["E", "A", "B", "C"]:
            self.batch_method.addItem(core.METHOD_NAMES[m], m)
        r.addWidget(self.batch_method)
        b = QPushButton("Обработать все и сохранить в папку…"); b.clicked.connect(self.run_batch)
        r.addWidget(b); r.addStretch(1); v.addLayout(r)
        self.batch_bar = QProgressBar(); v.addWidget(self.batch_bar)
        self.batch_table = QTableWidget(); v.addWidget(self.batch_table, 1)
        self.tabs.addTab(t, "5. Пакетная обработка")
        self.tabs.currentChanged.connect(lambda _: self.refresh_views())

    # ------------------------------------------------------- фоновые задачи
    def start_task(self, task, on_done, on_progress=None):
        """Результат задачи обрабатывается в главном потоке (через слоты окна)."""
        self._callbacks[id(task)] = on_done
        task.signals.done.connect(self._task_done)
        task.signals.error.connect(self.on_error)
        if on_progress is not None:
            self._progress_cb = on_progress
            task.signals.progress.connect(self._task_progress)
        self.tasks.append(task)
        self.pool.start(task)

    def _task_done(self, payload):
        tid, res = payload
        cb = self._callbacks.pop(tid, None)
        self.tasks = [t for t in self.tasks if id(t) != tid]
        if cb:
            cb(res)

    def _task_progress(self, i, text):
        if getattr(self, "_progress_cb", None):
            self._progress_cb(i, text)

    # ------------------------------------------------------- загрузка файлов
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls()]
        files = []
        for p in paths:
            if p.is_dir():
                files += sorted(q for q in p.iterdir() if q.suffix.lower() in core.IMG_EXT)
                if (p / "magnifications.csv").exists():
                    self.load_mags(p / "magnifications.csv")
            elif p.name.lower() == "magnifications.csv":
                self.load_mags(p)
            elif p.suffix.lower() in core.IMG_EXT:
                files.append(p)
        self.add_files(files)

    def open_files(self, *_):
        fs, _ = QFileDialog.getOpenFileNames(self, "Снимки", "",
                                             "Изображения (*.png *.tif *.tiff *.jpg *.jpeg *.bmp)")
        self.add_files([Path(f) for f in fs])

    def open_folder(self, *_):
        d = QFileDialog.getExistingDirectory(self, "Папка со снимками")
        if d:
            p = Path(d)
            if (p / "magnifications.csv").exists():
                self.load_mags(p / "magnifications.csv")
            self.add_files(sorted(q for q in p.iterdir() if q.suffix.lower() in core.IMG_EXT))

    def open_mags(self, *_):
        f, _ = QFileDialog.getOpenFileName(self, "magnifications.csv", "", "CSV (*.csv)")
        if f:
            self.load_mags(Path(f))

    def load_mags(self, path):
        new = core.read_magnifications(path)
        if not new:
            QMessageBox.warning(self, "magnifications.csv",
                                "В файле не найдено ни одного увеличения. Нужны колонки image и mag_kx "
                                "(разделитель — запятая, точка с запятой или табуляция).")
        self.mags.update(new)
        for st in self.images.values():
            st.result = None
        self.statusBar().showMessage(f"Загружено увеличений: {len(self.mags)}")
        self.update_scale_info()

    def add_files(self, files):
        first = None
        for p in files:
            if p.name in self.images:
                continue
            self.images[p.name] = ImageState(p)
            it = QListWidgetItem(p.name); self.list.addItem(it)
            first = first or it
        if first is not None:
            self.list.setCurrentItem(first)
        self.statusBar().showMessage(f"Снимков загружено: {len(self.images)}")

    def open_gt(self, *_):
        st = self.state()
        if not st:
            return
        f, _ = QFileDialog.getOpenFileName(self, f"Эталонная маска для {st.name}", "",
                                           "Изображения (*.png *.tif *.tiff *.bmp)")
        if f:
            st.gt_path = Path(f)
            self.gt_label.setText(f"Эталон: {st.gt_path.name}")
            self.tabs.setCurrentIndex(3)
            self.refresh_views()

    # ------------------------------------------------------- текущий снимок
    def state(self):
        return self.images.get(self.cur)

    def on_select(self, item, _prev=None):
        if item is None:
            return
        self.cur = item.text()
        st = self.state()
        self.scale_combo.blockSignals(True)
        self.scale_combo.setCurrentIndex({"auto": 0, "mag": 1, "px": 2}[st.scale_mode])
        self.scale_combo.blockSignals(False)
        self.on_scale_mode(self.scale_combo.currentIndex(), recompute=False)
        self.cut_check.blockSignals(True); self.cut_spin.blockSignals(True)
        self.cut_check.setChecked(st.cut_row is not None)
        self.cut_spin.setEnabled(st.cut_row is not None)
        if st.cut_row is not None:
            self.cut_spin.setValue(st.cut_row)
        self.cut_check.blockSignals(False); self.cut_spin.blockSignals(False)
        self.gt_label.setText(f"Эталон: {st.gt_path.name}" if st.gt_path else "Эталон не загружен")
        self.view._fitted = False
        if st.result is None:
            self.run_current()
        else:
            self.after_result()

    def on_scale_mode(self, idx, recompute=True):
        st = self.state()
        if not st:
            return
        try:
            self.scale_val.valueChanged.disconnect()
        except (RuntimeError, TypeError):
            pass
        st.scale_mode = ["auto", "mag", "px"][idx]
        self.scale_val.setEnabled(idx > 0)
        if idx == 1:
            self.scale_val.setValue(st.mag or self.mags.get(st.stem, 50.0))
        elif idx == 2:
            self.scale_val.setValue(st.px or 5.0)
        self.scale_val.valueChanged.connect(self.on_scale_value)
        if idx > 0:
            self.on_scale_value(self.scale_val.value())
        self.update_scale_info()

    def on_scale_value(self, v):
        st = self.state()
        if not st:
            return
        if st.scale_mode == "mag":
            st.mag = v
        elif st.scale_mode == "px":
            st.px = v
        self.scale_info.setText("Нажмите «Обработать снимок», чтобы пересчитать размеры.")

    def scale_args(self, st):
        if st.scale_mode == "mag":
            return st.mag, None
        if st.scale_mode == "px":
            return None, st.px
        return self.mags.get(st.stem), None

    def update_scale_info(self):
        st = self.state()
        if not st:
            return
        R = st.result
        if R is None:
            self.scale_info.setText("—")
        else:
            src = {"tiff_metadata": "метаданные TIFF", "magnification": "по увеличению",
                   "manual": "вручную", "unknown": "не задан — размеры в пикселях"}[R["scale_source"]]
            px = f"{R['pixel_nm']:.3f} нм/пикс." if R["pixel_nm"] else ""
            self.scale_info.setText(f"{px} ({src})")

    def on_cut_toggle(self, checked):
        st = self.state()
        if not st:
            return
        self.cut_spin.setEnabled(checked)
        if checked:
            R = st.result
            if R is not None:       # стартуем с того, что нашла автоматика
                v = R["cut_row"] if R["cut_source"] != "none" else int(0.9 * R["full_height"])
                self.cut_spin.blockSignals(True); self.cut_spin.setValue(v); self.cut_spin.blockSignals(False)
            st.cut_row = self.cut_spin.value()
        else:
            st.cut_row = None
        self.scale_info.setText("Нажмите «Обработать снимок», чтобы пересчитать с новой областью.")

    def on_cut_value(self, v):
        st = self.state()
        if st and self.cut_check.isChecked():
            st.cut_row = int(v)

    def merge_k(self):
        return {k: s.value() for k, s in self.mk.items()}

    # ---------------------------------------------------------------- расчёт
    def run_current(self, *_):
        st = self.state()
        if not st:
            return
        mag, px = self.scale_args(st)
        use_model = self.use_model.isChecked()
        key = (mag, px, use_model, tuple(sorted(self.merge_k().items())), st.cut_row)
        if st.result is not None and st.result_key == key:
            self.after_result(); return
        self.set_busy(True, f"Обработка {st.name}…")
        task = Task(core.compute, str(st.path), mag, px, use_model, self.merge_k(), cut_row=st.cut_row)
        name = st.name

        def done(R):
            s = self.images[name]
            s.result, s.result_key = R, key
            if s.method not in R["methods"]:
                s.method = "E"
            self.set_busy(False, f"{name}: готово")
            if name == self.cur:
                self.after_result()

        self.start_task(task, done)

    def set_busy(self, busy, msg=""):
        self.busy.setVisible(busy); self.btn_run.setEnabled(not busy)
        if msg:
            self.statusBar().showMessage(msg)

    def on_error(self, tb):
        self.set_busy(False, "Ошибка")
        QMessageBox.critical(self, "Ошибка", tb[-3000:])

    def after_result(self):
        st = self.state(); R = st.result
        for combo in (self.stage_method, self.edit_method):
            combo.blockSignals(True); combo.clear()
            for m in R["methods"]:
                combo.addItem(core.METHOD_NAMES[m], m)
            combo.blockSignals(False)
        self.stage_method.setCurrentIndex(R["methods"].index("E"))
        self.edit_method.setCurrentIndex(R["methods"].index(st.method))
        cs = {"auto": f"отрезана автоматически со строки {R['cut_row']}",
              "manual": f"отрезана вручную со строки {R['cut_row']}",
              "none": "не найдена, обрезка не выполнена"}[R["cut_source"]]
        warn = "".join(f"<br><span style='color:#b05c00'>⚠ {w}</span>" for w in R["warnings"])
        self.info.setText(
            f"<b>{st.name}</b><br>Рабочая область: {R['image'].shape[1]}×{R['image'].shape[0]} пикс.<br>"
            f"Плашка {cs}<br>"
            f"Размер зерна L ≈ {R['params']['L']:.1f} пикс.<br>"
            f"Согласие методов по границам: {R['agreement']:.0%}" + warn)
        self.update_scale_info()
        self.refresh_views()

    # ------------------------------------------------------------- вкладки
    def refresh_views(self):
        st = self.state()
        if not st or st.result is None:
            return
        i = self.tabs.currentIndex()
        if i == 0:
            self.draw_stages()
        elif i == 1:
            self.draw_edit()
        elif i == 2:
            self.draw_stats()
        elif i == 3:
            self.draw_quality()

    def final(self):
        st = self.state()
        return core.final_result(st.result, st.method, st.ops)

    def draw_stages(self, *_):
        st = self.state()
        if not st or st.result is None or self.stage_method.currentData() is None:
            return
        core.stages_figure(st.result, self.stage_method.currentData(), self.stage_canvas.fig)
        self.stage_canvas.draw_idle()

    def save_stages_png(self, *_):
        st = self.state()
        if not st or st.result is None:
            return
        m = self.stage_method.currentData()
        f, _ = QFileDialog.getSaveFileName(self, "Сохранить рисунок", f"{st.stem}_stages_{m}.png",
                                           "PNG (*.png)")
        if f:
            self.stage_canvas.fig.savefig(f, dpi=110)

    def on_edit_method(self, *_):
        st = self.state()
        if st and self.edit_method.currentData():
            st.method = self.edit_method.currentData()
            self.draw_edit()

    def draw_edit(self, *_):
        st = self.state()
        if not st or st.result is None:
            return
        fin = self.final()
        hl = None
        if st.pending:
            gid = ed._id_at(fin["lab"], st.pending)
            hl = {gid} if gid else None
        zoom = 3 if st.result["image"].shape[1] < 600 else 1
        view = ed.render(st.result["image"], fin["lab"], fin["cm"],
                         show_ids=self.show_ids.isChecked(), scale=zoom, highlight=hl)
        self._zoom = zoom
        self.view.set_image(view)
        self.log.setPlainText("\n".join(fin["log"]) or "Правок пока нет.")

    def on_click(self, y, x):
        st = self.state()
        if not st or st.result is None:
            return
        y, x = y // self._zoom, x // self._zoom
        p = [int(y), int(x)]
        act = self.actions.checkedId()
        fin = self.final()
        gid = ed._id_at(fin["lab"], p)
        if act == 0:
            if gid:
                g = next((g for g in fin["grains"] if g["id"] == gid), None)
                unit = "нм" if st.result["pixel_nm"] else "пикс."
                d = g["eq_diameter_nm"] if st.result["pixel_nm"] else np.sqrt(4 * g["area_px"] / np.pi)
                self.edit_hint.setText(f"Объект №{gid}: класс «{g['class']}», "
                                       f"диаметр {d:.0f} {unit}, эксцентриситет {g['eccentricity']:.2f}")
            else:
                self.edit_hint.setText("Здесь граница или фон.")
            return
        if act == 1:
            st.ops.append({"op": "delete", "p": p})
        elif act == 4:
            st.ops.append({"op": "class", "p": p, "cls": self.cls_combo.currentData()})
        else:
            op = "merge" if act == 2 else "split"
            if st.pending is None:
                st.pending = p
                self.edit_hint.setText("Первая точка выбрана (подсвечено). Кликните вторую.")
                self.draw_edit(); return
            st.ops.append({"op": op, "p": st.pending, "q": p})
            st.pending = None
        self.edit_hint.setText("")
        self.draw_edit()

    def cancel_pending(self):
        st = self.state()
        if st and st.pending:
            st.pending = None; self.edit_hint.setText(""); self.draw_edit()

    def undo(self, *_):
        st = self.state()
        if st and st.ops:
            st.ops.pop(); st.pending = None; self.draw_edit()

    def reset_ops(self, *_):
        st = self.state()
        if st and st.ops and QMessageBox.question(self, "Сброс", "Удалить все правки этого снимка?") \
                == QMessageBox.Yes:
            st.ops = []; st.pending = None; self.draw_edit()

    def save_ops(self, *_):
        st = self.state()
        if not st:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Сохранить правки", f"{st.stem}_corrections.json",
                                           "JSON (*.json)")
        if f:
            Path(f).write_text(ed.corrections_json(st.name, st.method, st.ops), encoding="utf-8")
            self.statusBar().showMessage(f"Правки сохранены: {f}")

    def load_ops(self, *_):
        st = self.state()
        if not st:
            return
        f, _ = QFileDialog.getOpenFileName(self, "Загрузить правки", "", "JSON (*.json)")
        if f:
            ops, meta = ed.load_corrections(Path(f).read_text(encoding="utf-8"))
            st.ops = ops
            if meta.get("method") in (st.result or {}).get("methods", []):
                st.method = meta["method"]
                self.edit_method.setCurrentIndex(st.result["methods"].index(st.method))
            self.draw_edit()
            self.statusBar().showMessage(f"Загружено правок: {len(ops)}")

    def draw_stats(self):
        st = self.state()
        fin = self.final()
        s = core.grain_stats(fin["grains"], st.result["pixel_nm"])
        self.stats_label.setText(
            f"Разметка: метод {fin['method']} с учётом правок ({len(st.ops)}). "
            f"Целых зёрен: <b>{s['n']}</b> из {len(fin['grains'])} объектов. "
            f"Медианный диаметр: <b>{s['median']:.0f} {s['unit']}</b>, "
            f"среднее ± СКО: {s['mean']:.0f} ± {s['std']:.0f} {s['unit']}, "
            f"средний эксцентриситет: {s['ecc']:.2f}")
        fig = self.hist_canvas.fig; fig.clear()
        a1, a2 = fig.subplots(1, 2)
        if s["n"]:
            a1.hist(s["d"], bins=25, color="#c0392b", alpha=0.85)
            a2.hist(s["circ"], bins=25, color="#2c7fb8", alpha=0.85)
        a1.set_xlabel(f"Эквивалентный диаметр, {s['unit']}"); a1.set_ylabel("Число зёрен")
        a2.set_xlabel("Округлость 4πS/P²"); a2.set_ylabel("Число зёрен")
        fig.tight_layout(); self.hist_canvas.draw_idle()
        self.grain_table.setSortingEnabled(False)
        fill_table(self.grain_table, fin["grains"])
        self.grain_table.setSortingEnabled(True)

    def save_grains_csv(self, *_):
        st = self.state()
        if not st or st.result is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Таблица зёрен", f"{st.stem}_grains.csv", "CSV (*.csv)")
        if f:
            Path(f).write_bytes(ed.csv_bytes(self.final()["grains"]))

    def draw_quality(self):
        st = self.state()
        if not st.gt_path:
            fill_table(self.q_table, []); return
        R = st.result; img = R["image"]
        gt = load_gt(st.gt_path, img.shape)
        valid = annotated_region(gt)
        rows = []
        for m in R["methods"]:
            rows.append({"метод": core.METHOD_NAMES[m], **evaluate(gt, R[m]["post"], valid=valid)})
        fin = self.final()
        rows.append({"метод": f"{fin['method']} + ручные правки", **evaluate(gt, fin["lab"], valid=valid)})
        fill_table(self.q_table, rows)
        from skimage import segmentation
        rgb = segmentation.mark_boundaries(img, np.maximum(gt, 0), color=(0, 1, 0))
        rgb = segmentation.mark_boundaries(rgb, fin["lab"], color=(1, 0, 0))
        rgb[~valid] *= 0.45
        self.q_view.set_image(rgb, keep_view=False)

    # ------------------------------------------------------------ сохранение
    def save_current(self, *_):
        st = self.state()
        if not st or st.result is None:
            return
        d = QFileDialog.getExistingDirectory(self, "Папка для результатов")
        if d:
            od = core.export_one(d, st.stem, st.result, self.final(), st.ops)
            self.statusBar().showMessage(f"Сохранено в {od}")

    def run_batch(self, *_):
        if not self.images:
            return
        d = QFileDialog.getExistingDirectory(self, "Папка для результатов пакетной обработки")
        if not d:
            return
        method = self.batch_method.currentData()
        use_model = self.use_model.isChecked()
        mk = self.merge_k()
        jobs = []
        for st in self.images.values():
            mag, px = self.scale_args(st)
            jobs.append((st.name, st.stem, str(st.path), mag, px, list(st.ops), st.cut_row))
        if method == "C" and not (use_model and core.MODEL_PATH.exists()):
            method = "E"

        def work(progress):
            rows = []
            for i, (name, stem, path, mag, px, ops, cut) in enumerate(jobs):
                progress(i, name)
                R = core.compute(path, mag, px, use_model, mk, cut_row=cut)
                fin = core.final_result(R, method, ops)
                core.export_one(d, stem, R, fin, ops)
                rows.append(core.summary_row(stem, R, fin))
            out = Path(d) / "summary.csv"
            Path(out).write_bytes(ed.csv_bytes(rows))
            progress(len(jobs), "готово")
            return rows

        self.batch_bar.setRange(0, len(jobs)); self.batch_bar.setValue(0)
        def prog(i, n):
            self.batch_bar.setValue(i); self.statusBar().showMessage(f"Пакетная обработка: {n}")

        def finished(rows):
            fill_table(self.batch_table, rows)
            self.statusBar().showMessage(f"Готово, результаты и summary.csv в {d}")

        self.start_task(Task(work, with_progress=True), finished, prog)

    def show_help(self, *_):
        QMessageBox.information(self, "Как пользоваться", (
            "1. Перетащите снимки (или папку) в окно. Если есть magnifications.csv — "
            "перетащите и его: он задаёт увеличение для PNG.\n"
            "2. Выберите снимок в списке слева — он обработается автоматически. "
            "После смены масштаба или параметров нажмите «Обработать снимок».\n"
            "3. «Стадии обработки» — 6 стадий для каждого метода.\n"
            "4. «Ручная правка» — выберите действие и кликайте по зёрнам. "
            "Колесо мыши — масштаб, правая кнопка — перетаскивание.\n"
            "5. «Характеристики зёрен» — таблица и гистограммы с учётом правок.\n"
            "6. «Качество по эталону» — загрузите ручную маску, чтобы увидеть метрики.\n"
            "7. «Пакетная обработка» — все снимки сразу, результаты в выбранную папку.\n"
            "Ctrl+S — сохранить все результаты текущего снимка."))


def main():
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    args = [Path(a) for a in sys.argv[1:]]
    if args:
        w.add_files([a for a in args if a.suffix.lower() in core.IMG_EXT])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()