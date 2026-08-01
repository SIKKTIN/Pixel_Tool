"""
Perfect Pixel Tool — 本地桌面应用 (PySide6)

启动:  python desktop_app.py
打包:  pyinstaller --noconsole --windowed --onefile --name PerfectPixelTool desktop_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QAction, QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from perfect_pixel import get_perfect_pixel


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """H x W x 3 uint8 -> QPixmap。"""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def resize_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape[:2]
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


# ---------------------------------------------------------------------------
# 后台工作线程
# ---------------------------------------------------------------------------

class RefineWorker(QThread):
    finished_ok = Signal(object, object, object)  # w, h, ndarray
    failed = Signal(str)

    def __init__(self, image: np.ndarray, sample_method: str,
                 refine_intensity: float, fix_square: bool) -> None:
        super().__init__()
        self.image = image
        self.sample_method = sample_method
        self.refine_intensity = refine_intensity
        self.fix_square = fix_square

    def run(self) -> None:
        try:
            w, h, out = get_perfect_pixel(
                self.image,
                sample_method=self.sample_method,
                refine_intensity=self.refine_intensity,
                fix_square=self.fix_square,
                debug=False,
            )
            if w is None or h is None or out is None:
                self.failed.emit("未能从图片中检测到像素网格,请换一张更明显的像素风图。")
                return
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
            self.finished_ok.emit(w, h, out)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# 像素细化工具页
# ---------------------------------------------------------------------------

class PixelRefineWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.input_image: np.ndarray | None = None  # 原始输入(RGB uint8)
        self.output_image: np.ndarray | None = None  # 像素化结果
        self.last_saved_path: str | None = None  # 上次保存路径
        self.worker: RefineWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- 顶部控件栏 --------------------------------------------------
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.btn_open = QPushButton("打开图片…")
        self.btn_open.clicked.connect(self.on_open)
        ctrl_row.addWidget(self.btn_open)

        self.btn_save = QPushButton("保存结果…")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)
        ctrl_row.addWidget(self.btn_save)

        ctrl_row.addSpacing(20)

        ctrl_row.addWidget(QLabel("采样:"))
        self.cmb_sample = QComboBox()
        self.cmb_sample.addItems(["center", "median", "majority"])
        ctrl_row.addWidget(self.cmb_sample)

        ctrl_row.addWidget(QLabel("对齐强度:"))
        self.sld_intensity = QSlider(Qt.Horizontal)
        self.sld_intensity.setRange(0, 50)
        self.sld_intensity.setValue(30)
        self.sld_intensity.setFixedWidth(140)
        ctrl_row.addWidget(self.sld_intensity)
        self.lbl_intensity = QLabel("0.30")
        ctrl_row.addWidget(self.lbl_intensity)
        self.sld_intensity.valueChanged.connect(
            lambda v: self.lbl_intensity.setText(f"{v / 100:.2f}")
        )

        self.chk_square = QCheckBox("强制正方形")
        self.chk_square.setChecked(True)
        ctrl_row.addWidget(self.chk_square)

        ctrl_row.addWidget(QLabel("预览倍数:"))
        self.spn_scale = QSpinBox()
        self.spn_scale.setRange(2, 32)
        self.spn_scale.setValue(8)
        ctrl_row.addWidget(self.spn_scale)

        self.btn_run = QPushButton("生成像素图")
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self.on_run)
        ctrl_row.addWidget(self.btn_run)

        ctrl_row.addStretch(1)
        root.addLayout(ctrl_row)

        # ---- 三列预览区 --------------------------------------------------
        preview_row = QHBoxLayout()
        preview_row.setSpacing(10)

        self.view_input = ImageView("原图")
        self.view_output = ImageView("像素化结果")
        self.view_preview = ImageView("放大预览")
        for v in (self.view_input, self.view_output, self.view_preview):
            preview_row.addWidget(v, 1)
        root.addLayout(preview_row, 1)

        # ---- 拖拽上传 ----------------------------------------------------
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    # 拖拽支持
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        local = urls[0].toLocalFile()
        if local:
            self.load_path(local)

    # ------------------------------------------------------------------
    # 槽函数
    # ------------------------------------------------------------------
    def on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*.*)",
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "打开失败", str(exc))
            return
        self.input_image = np.array(img)
        self.view_input.set_image(self.input_image)
        self.view_output.clear()
        self.view_preview.clear()
        self.output_image = None
        self.btn_save.setEnabled(False)
        self.last_saved_path = None

    def on_run(self) -> None:
        if self.input_image is None:
            QMessageBox.information(self, "提示", "请先打开一张图片")
            return
        if self.worker is not None and self.worker.isRunning():
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("处理中…")

        self.worker = RefineWorker(
            image=self.input_image,
            sample_method=self.cmb_sample.currentText(),
            refine_intensity=self.sld_intensity.value() / 100,
            fix_square=self.chk_square.isChecked(),
        )
        self.worker.finished_ok.connect(self.on_refine_done)
        self.worker.failed.connect(self.on_refine_failed)
        self.worker.start()

    def on_refine_done(self, w: int, h: int, out: np.ndarray) -> None:
        self.output_image = out
        self.view_output.set_image(out)
        self.view_preview.set_image(resize_nearest(out, self.spn_scale.value()))
        self.btn_run.setEnabled(True)
        self.btn_run.setText("生成像素图")
        self.btn_save.setEnabled(True)
        self.status_message(
            f"网格 {w} × {h}  |  预览 {self.spn_scale.value()}×  "
            f"|  采样 {self.cmb_sample.currentText()}"
        )

    def on_refine_failed(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("生成像素图")
        QMessageBox.warning(self, "处理失败", msg)

    def on_save(self) -> None:
        if self.output_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果",
            "output_native.png",
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if not path:
            return
        try:
            Image.fromarray(self.output_image).save(path)
            self.last_saved_path = path
            self.status_message(f"已保存到 {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(exc))

    # ------------------------------------------------------------------
    # 状态栏
    # ------------------------------------------------------------------
    def status_message(self, text: str) -> None:
        win = self.window()
        if isinstance(win, QMainWindow):
            sb = win.statusBar()
            if sb is not None:
                sb.showMessage(text, 5000)


# ---------------------------------------------------------------------------
# 可缩放图片视图
# ---------------------------------------------------------------------------

class ImageView(QWidget):
    """一张可以拖拽 / 缩放 / 适应窗口的图像预览面板。"""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QScrollArea

        self._pixmap: QPixmap | None = None
        self._scale: float = 1.0

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        header = QHBoxLayout()
        self.title_label = QLabel(title)
        header.addWidget(self.title_label)
        header.addStretch(1)

        self.btn_fit = QPushButton("适应")
        self.btn_fit.setFixedWidth(56)
        self.btn_fit.clicked.connect(self.fit_to_view)
        header.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("100%")
        self.btn_100.setFixedWidth(56)
        self.btn_100.clicked.connect(self.actual_size)
        header.addWidget(self.btn_100)

        v.addLayout(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll.setWidget(self._label)
        v.addWidget(self._scroll, 1)

        self._placeholder = QLabel("(无图片)")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self._label.setText("(无图片)")

    # ------------------------------------------------------------------
    def set_image(self, arr: np.ndarray) -> None:
        self._pixmap = numpy_to_qpixmap(arr)
        self._apply()

    def clear(self) -> None:
        self._pixmap = None
        self._label.setPixmap(QPixmap())
        self._label.setText("(无图片)")
        self._label.resize(self.size())

    def fit_to_view(self) -> None:
        if self._pixmap is None:
            return
        view_size = self._scroll.viewport().size()
        if view_size.width() <= 0 or view_size.height() <= 0:
            return
        sx = view_size.width() / self._pixmap.width()
        sy = view_size.height() / self._pixmap.height()
        self._scale = max(0.05, min(sx, sy))
        self._apply()

    def actual_size(self) -> None:
        if self._pixmap is None:
            return
        self._scale = 1.0
        self._apply()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # 保持响应式:首次获得尺寸时自动适应
        if self._pixmap is not None and self._scale == 0.0:
            self.fit_to_view()

    # ------------------------------------------------------------------
    def _apply(self) -> None:
        if self._pixmap is None:
            return
        size = self._pixmap.size()
        scaled = self._pixmap.scaled(
            int(size.width() * self._scale),
            int(size.height() * self._scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())

    def wheelEvent(self, event) -> None:
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1 / 1.25
        self._scale = max(0.05, min(20.0, self._scale * factor))
        self._apply()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Perfect Pixel Tool")
        self.resize(1280, 800)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)

        # ------- 第一个工具 -------
        self.pixel_tab = PixelRefineWidget()
        self.tabs.addTab(self.pixel_tab, "🎨 像素细化")

        # ------- 预留 Tab: 后续添加工具的位置 -------
        placeholder = QWidget()
        ph_layout = QVBoxLayout(placeholder)
        ph_layout.setAlignment(Qt.AlignCenter)
        ph_label = QLabel("🚧 工具开发中…\n\n下一个工具(如批量处理 / 颜色量化 / 动画导出等)\n会在此添加。")
        ph_label.setAlignment(Qt.AlignCenter)
        ph_label.setStyleSheet("color: #888; font-size: 16px;")
        ph_layout.addWidget(ph_label)
        self.tabs.addTab(placeholder, "➕ 即将到来")

        self.setCentralWidget(self.tabs)

        # ------- 工具栏快捷键 -------
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        action_open = QAction("打开", self)
        action_open.setShortcut(QKeySequence.Open)
        action_open.triggered.connect(self.pixel_tab.on_open)
        toolbar.addAction(action_open)

        action_save = QAction("保存", self)
        action_save.setShortcut(QKeySequence.Save)
        action_save.triggered.connect(self.pixel_tab.on_save)
        toolbar.addAction(action_save)

        # ------- 状态栏 -------
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("就绪 — 拖拽图片到窗口,或点击「打开图片」")

        # ------- Ctrl+W 关闭当前 Tab(disabled:有标签页模型)-------
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

    def register_tab(self, widget: QWidget, title: str) -> None:
        """未来新增工具时调用:self.register_tab(NewToolWidget(), '去水印')"""
        self.tabs.insertTab(self.tabs.count() - 1, widget, title)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PerfectPixelTool")
    app.setOrganizationName("PerfectPixelTool")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
