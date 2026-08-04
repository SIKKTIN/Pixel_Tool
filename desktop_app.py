"""
Perfect Pixel Tool — 本地桌面应用 (PySide6)

启动:  python desktop_app.py
打包:  pyinstaller --noconsole --windowed --onefile --name PerfectPixelTool desktop_app.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal, QSize, QObject
from PySide6.QtGui import QAction, QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
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
    QSplitter,
)


# ============================================================
# 图片暂存区（中央管理器）
# ============================================================

class ImageBuffer(QObject):
    """全局图片暂存管理器，供所有 Tab 共享。

    存储每个暂存项的：ID、numpy 数组、来源 Tab、创建时间、来源文件名。
    右侧面板监听此对象的变化来更新缩略图。
    """

    changed = Signal()

    def __init__(self, max_items: int = 20):
        super().__init__()
        self._items: list[dict] = []
        self._max_items = max_items
        self._active_id: Optional[str] = None

    def push(self, image: np.ndarray, source_tab: str = "", source_file: str = "") -> str:
        """把一张图片压入暂存区，返回新项 ID。"""
        item_id = uuid.uuid4().hex[:8]
        h, w = image.shape[:2]
        self._items.append({
            "id": item_id,
            "image": image,
            "source_tab": source_tab,
            "source_file": source_file,
            "size_text": f"{w} × {h}",
            "created_at": len(self._items) + 1,
        })
        if len(self._items) > self._max_items:
            self._items.pop(0)
        self._active_id = item_id
        self.changed.emit()
        return item_id

    def set_active(self, item_id: str) -> None:
        """把指定项设为当前选中。"""
        if any(it["id"] == item_id for it in self._items):
            self._active_id = item_id
            self.changed.emit()

    def get_active(self) -> Optional[np.ndarray]:
        """获取当前选中的图片数组，没有则返回 None。"""
        for it in self._items:
            if it["id"] == self._active_id:
                return it["image"]
        return None

    def get_by_id(self, item_id: str) -> Optional[np.ndarray]:
        for it in self._items:
            if it["id"] == item_id:
                return it["image"]
        return None

    def remove(self, item_id: str) -> None:
        self._items = [it for it in self._items if it["id"] != item_id]
        if self._active_id == item_id:
            self._active_id = self._items[-1]["id"] if self._items else None
        self.changed.emit()

    def clear(self) -> None:
        self._items.clear()
        self._active_id = None
        self.changed.emit()

    def items(self) -> list[dict]:
        return list(self._items)

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id


# 全局唯一实例
_image_buffer: Optional[ImageBuffer] = None


def image_buffer() -> ImageBuffer:
    global _image_buffer
    if _image_buffer is None:
        _image_buffer = ImageBuffer()
    return _image_buffer


# ============================================================
# 右侧暂存区面板
# ============================================================

class ImageTrayWidget(QWidget):
    """悬浮在主窗口右侧的图片暂存区。

    行为:
    - 显示缩略图列表，最新的在最上（栈式）
    - 点击缩略图 → 设为当前选中（高亮边框）
    - 双击缩略图 → 把图片送回当前 Tab 作为输入（发送 load_request 信号）
    - 工具栏:清除全部 / 删除选中
    - 空白时显示提示文字
    """

    load_request = Signal(str)   # item_id

    THUMB_SIZE = 120
    COLUMNS = 2

    def __init__(self, buffer: ImageBuffer, parent: QWidget | None = None):
        super().__init__(parent)
        self._buf = buffer
        self._thumb_widgets: dict[str, QWidget] = {}

        self.setFixedWidth(self.THUMB_SIZE * self.COLUMNS + 16)
        self.setMinimumHeight(200)
        self.setMaximumWidth(400)

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 4, 2, 4)
        root.setSpacing(6)

        # --- 标题栏 ---
        header = QHBoxLayout()
        lbl = QLabel("📦 暂存区")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        header.addWidget(lbl)
        header.addStretch(1)

        self._count_label = QLabel("0")
        self._count_label.setStyleSheet("color: #888; font-size: 12px;")
        header.addWidget(self._count_label)

        self.btn_clear = QPushButton("清空")
        self.btn_clear.setFixedSize(40, 22)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_clear.setStyleSheet("font-size: 11px; padding: 0;")
        header.addWidget(self.btn_clear)
        root.addLayout(header)

        # --- 缩略图网格容器 ---
        self._row_widget = QWidget()
        self._grid = QVBoxLayout(self._row_widget)
        self._grid.setSpacing(6)
        self._grid.addStretch(1)
        root.addWidget(_TrayScrollArea(self._row_widget, self))

        # --- 底部:选中图信息 ---
        self._info_label = QLabel("双击缩略图送入当前模块")
        self._info_label.setStyleSheet("color: #666; font-size: 11px;")
        self._info_label.setWordWrap(True)
        root.addWidget(self._info_label)

        buffer.changed.connect(self._refresh)

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        """整体重建缩略图列表（项不多，重建开销可接受）。"""
        # 先断开信号，避免重建过程中触发的 changed 信号导致递归
        try:
            self._buf.changed.disconnect(self._refresh)
        except RuntimeError:
            pass

        # 清除所有已有 row widgets（保留最后的 stretch）
        while self._grid.count() > 1:
            child = self._grid.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                child.layout().deleteLater()
            del child

        items = list(reversed(self._buf.items()))
        self._count_label.setText(str(len(items)))
        self._thumb_widgets.clear()

        for row_start in range(0, len(items), self.COLUMNS):
            row_items = items[row_start:row_start + self.COLUMNS]
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setSpacing(6)
            row_layout.setContentsMargins(0, 0, 10, 0)
            for item in row_items:
                w = _ThumbItem(item, self._buf.active_id, self)
                w.clicked.connect(self._on_thumb_click)
                w.double_clicked.connect(self._on_thumb_dclick)
                self._thumb_widgets[item["id"]] = w
                row_layout.addWidget(w)
            # 不足一列时用空白 widget 填充，勿用 stretch（会撑大整行）
            if len(row_items) < self.COLUMNS:
                spacer = QWidget()
                row_layout.addWidget(spacer)
            row_widget.setFixedWidth(
                self.THUMB_SIZE * self.COLUMNS + (self.COLUMNS - 1) * 6
            )
            # 插入到 stretch 之前
            self._grid.insertWidget(self._grid.count() - 1, row_widget)

        # 更新 info
        active = self._buf.get_active()
        if active is None:
            self._info_label.setText("双击缩略图送入当前模块")
            self._info_label.setStyleSheet("color: #666; font-size: 11px;")
        else:
            self._info_label.setText("当前选中，已高亮")
            self._info_label.setStyleSheet("color: #4a9; font-size: 11px;")

        self._buf.changed.connect(self._refresh)

    def _on_thumb_click(self, item_id: str) -> None:
        self._buf.set_active(item_id)

    def _on_thumb_dclick(self, item_id: str) -> None:
        self._buf.set_active(item_id)
        self.load_request.emit(item_id)

    def _on_clear(self) -> None:
        self._buf.clear()


class _TrayScrollArea(QWidget):
    """带滚动条的缩略图网格容器。"""

    def __init__(self, content_widget: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("border: none; background: transparent;")
        scroll.viewport().setStyleSheet("border: none; margin: 0; padding: 0;")
        scroll.setWidget(content_widget)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)


class _ThumbItem(QWidget):
    """单张缩略图卡片:点击选中，双击触发 load。"""

    clicked = Signal(str)
    double_clicked = Signal(str)

    THUMB = ImageTrayWidget.THUMB_SIZE
    PADDING = 4

    def __init__(self, item: dict, active_id: str | None, parent: QWidget | None = None):
        super().__init__(parent)
        self._item = item
        self._item_id = item["id"]
        self._is_active = item["id"] == active_id

        self.setFixedSize(self.THUMB, self.THUMB + 28)
        self.setMinimumSize(self.THUMB, self.THUMB + 28)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"来源: {item['source_tab']}\n尺寸: {item['size_text']}")

        self._lbl_img = QLabel(self)
        self._lbl_img.setFixedSize(self.THUMB - 2, self.THUMB - 2)
        self._lbl_img.setAlignment(Qt.AlignCenter)
        self._lbl_img.setStyleSheet("background: #2a2a2a;")
        self._lbl_img.move(1, 1)

        self._lbl_size = QLabel(item["size_text"], self)
        self._lbl_size.setAlignment(Qt.AlignCenter)
        self._lbl_size.setStyleSheet("font-size: 10px; color: #888; background: transparent; border: none;")
        self._lbl_size.setFixedWidth(self.THUMB - 2)
        self._lbl_size.move(1, self.THUMB)

        self._set_thumbnail(item["image"])
        self._update_border()

    def _set_thumbnail(self, arr: np.ndarray) -> None:
        thumb_h = self.THUMB
        h, w = arr.shape[:2]
        scale = thumb_h / max(h, w)
        tw, th = int(w * scale), int(h * scale)
        img = Image.fromarray(arr)
        img_small = img.resize((tw, th), Image.LANCZOS)
        qimg = QImage(img_small.tobytes(), tw, th, 3 * tw, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._lbl_img.setPixmap(pix)

    def _update_border(self) -> None:
        if self._is_active:
            self.setStyleSheet(
                "border: 2px solid #4caf50; border-radius: 6px; "
                "background: #1e3a1e;"
            )
        else:
            self.setStyleSheet(
                "border: 1px solid #444; border-radius: 6px; "
                "background: #252525;"
            )

    def enterEvent(self, event) -> None:
        if not self._is_active:
            self.setStyleSheet(
                "border: 1px solid #666; border-radius: 6px; "
                "background: #303030;"
            )
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._update_border()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._is_active = True
            self._update_border()
            self.clicked.emit(self._item_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self._item_id)
        super().mouseDoubleClickEvent(event)


# 把 src/ 加入 sys.path，使 watermark_remover 子包和 perfect_pixel 可直接 import
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

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
        scale = self.spn_scale.value()
        preview = resize_nearest(out, scale)
        self.view_preview.set_image(preview)
        self.view_preview.info_label.setText(f"{w * scale} × {h * scale} px ({scale}×)")
        self.btn_run.setEnabled(True)
        self.btn_run.setText("生成像素图")
        self.btn_save.setEnabled(True)
        image_buffer().push(out, source_tab="像素细化")
        self.status_message(
            f"网格 {w} × {h}  |  预览 {scale}×  "
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

    def load_from_buffer(self, image: np.ndarray) -> None:
        """从暂存区双击接收一张图片，作为新的输入原图。"""
        self.input_image = image
        self.view_input.set_image(image)
        self.view_output.clear()
        self.view_preview.clear()
        self.output_image = None
        self.btn_save.setEnabled(False)
        self.last_saved_path = None
        h, w = image.shape[:2]
        self.status_message(f"已从暂存区加载 ({w} × {h})")

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

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #c33; font-weight: bold; font-size: 12px;")
        header.addWidget(self.info_label)

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
        h, w = arr.shape[:2]
        self.info_label.setText(f"{w} × {h} px")
        self._apply()

    def clear(self) -> None:
        self._pixmap = None
        self._label.setPixmap(QPixmap())
        self._label.setText("(无图片)")
        self._label.resize(self.size())
        self.info_label.setText("")

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
# 尺寸缩放工具页
# ---------------------------------------------------------------------------

# 算法名 -> cv2 插值标志
SCALE_ALGORITHMS: dict[str, int] = {
    "最近邻 (Nearest)": cv2.INTER_NEAREST,
    "双线性 (Bilinear)": cv2.INTER_LINEAR,
    "双三次 (Bicubic)": cv2.INTER_CUBIC,
    "Lanczos": cv2.INTER_LANCZOS4,
    "区域平均 (Area)": cv2.INTER_AREA,
}


class ScaleWidget(QWidget):
    """尺寸缩放工具:打开原图,通过算法 + 目标尺寸实时缩放,支持导出。"""

    _buf: "ImageBuffer | None" = None

    @staticmethod
    def set_buffer_ref(buf: "ImageBuffer | None") -> None:
        ScaleWidget._buf = buf

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.input_image: np.ndarray | None = None  # 原图 RGB uint8
        self.last_saved_path: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- 顶部控件栏 --------------------------------------------------
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.btn_open = QPushButton("打开图片…")
        self.btn_open.clicked.connect(self.on_open)
        ctrl_row.addWidget(self.btn_open)

        self.lbl_file = QLabel("未选择文件")
        self.lbl_file.setStyleSheet("color: #666;")
        ctrl_row.addWidget(self.lbl_file, 1)

        ctrl_row.addStretch(1)
        root.addLayout(ctrl_row)

        # ---- 左右分栏:参数 + 预览 -----------------------------------------
        body = QHBoxLayout()
        body.setSpacing(10)

        # 左:参数面板
        param_box = QGroupBox("缩放参数")
        form = QFormLayout(param_box)
        form.setContentsMargins(10, 14, 10, 10)
        form.setSpacing(8)

        self.cmb_algo = QComboBox()
        self.cmb_algo.addItems(list(SCALE_ALGORITHMS.keys()))
        self.cmb_algo.setCurrentText("最近邻 (Nearest)")
        self.cmb_algo.currentTextChanged.connect(self._refresh_preview)
        form.addRow("插值算法:", self.cmb_algo)

        self.spn_w = QSpinBox()
        self.spn_w.setRange(1, 16384)
        self.spn_w.setValue(1)
        self.spn_w.setFixedWidth(120)
        form.addRow("目标宽度:", self.spn_w)

        self.spn_h = QSpinBox()
        self.spn_h.setRange(1, 16384)
        self.spn_h.setValue(1)
        self.spn_h.setFixedWidth(120)
        form.addRow("目标高度:", self.spn_h)

        self.chk_ratio = QCheckBox("保持宽高比")
        self.chk_ratio.setChecked(True)
        form.addRow("", self.chk_ratio)

        self.dsp_scale = QDoubleSpinBox()
        self.dsp_scale.setRange(0.01, 100.0)
        self.dsp_scale.setDecimals(2)
        self.dsp_scale.setSingleStep(0.1)
        self.dsp_scale.setValue(1.0)
        self.dsp_scale.setFixedWidth(120)
        form.addRow("缩放倍率:", self.dsp_scale)

        self.btn_apply = QPushButton("应用倍率")
        self.btn_apply.clicked.connect(self._apply_scale_to_size)
        form.addRow("", self.btn_apply)

        body.addWidget(param_box, 0)

        # 右:双图预览
        preview_wrap = QVBoxLayout()
        preview_wrap.setSpacing(10)
        self.view_input = ImageView("原图")
        self.view_output = ImageView("缩放后预览")
        preview_wrap.addWidget(self.view_input, 1)
        preview_wrap.addWidget(self.view_output, 1)
        preview_wrap_w = QWidget()
        preview_wrap_w.setLayout(preview_wrap)
        body.addWidget(preview_wrap_w, 1)
        root.addLayout(body, 1)

        # ---- 底部导出栏 --------------------------------------------------
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.btn_save_png = QPushButton("导出 PNG…")
        self.btn_save_png.clicked.connect(lambda: self._on_save("png"))
        self.btn_save_png.setEnabled(False)
        bottom.addWidget(self.btn_save_png)

        self.btn_save_jpg = QPushButton("导出 JPEG…")
        self.btn_save_jpg.clicked.connect(lambda: self._on_save("jpg"))
        self.btn_save_jpg.setEnabled(False)
        bottom.addWidget(self.btn_save_jpg)
        bottom.addStretch(1)
        root.addLayout(bottom)

        # ---- 拖拽上传 ----------------------------------------------------
        self.setAcceptDrops(True)

        # 信号连接(放在最末,避免 setValue 触发空刷新)
        self.spn_w.valueChanged.connect(self._on_w_changed)
        self.spn_h.valueChanged.connect(self._on_h_changed)
        self.dsp_scale.valueChanged.connect(self._on_scale_changed)

    # ------------------------------------------------------------------
    # 拖拽
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
    # 数据流
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
        h, w = self.input_image.shape[:2]
        self.lbl_file.setText(Path(path).name)
        self.view_input.set_image(self.input_image)
        # 初始化目标尺寸 = 原图尺寸
        self.spn_w.blockSignals(True)
        self.spn_h.blockSignals(True)
        self.spn_w.setValue(w)
        self.spn_h.setValue(h)
        self.spn_w.blockSignals(False)
        self.spn_h.blockSignals(False)
        self.dsp_scale.setValue(1.0)
        self._refresh_preview()
        self.btn_save_png.setEnabled(True)
        self.btn_save_jpg.setEnabled(True)
        self.status_message(f"已加载 {Path(path).name} ({w} × {h})")

    # ------------------------------------------------------------------
    # 信号回调
    # ------------------------------------------------------------------
    def _on_w_changed(self, w: int) -> None:
        if self.input_image is None:
            return
        h0, w0 = self.input_image.shape[:2]
        self.dsp_scale.blockSignals(True)
        if w0:
            self.dsp_scale.setValue(w / w0)
        self.dsp_scale.blockSignals(False)
        if self.chk_ratio.isChecked() and w0:
            new_h = max(1, round(w * h0 / w0))
            if new_h != self.spn_h.value():
                self.spn_h.blockSignals(True)
                self.spn_h.setValue(new_h)
                self.spn_h.blockSignals(False)
        self._refresh_preview()

    def _on_h_changed(self, h: int) -> None:
        if self.input_image is None:
            return
        h0, w0 = self.input_image.shape[:2]
        self.dsp_scale.blockSignals(True)
        if h0:
            self.dsp_scale.setValue(h / h0)
        self.dsp_scale.blockSignals(False)
        if self.chk_ratio.isChecked() and h0:
            new_w = max(1, round(h * w0 / h0))
            if new_w != self.spn_w.value():
                self.spn_w.blockSignals(True)
                self.spn_w.setValue(new_w)
                self.spn_w.blockSignals(False)
        self._refresh_preview()

    def _on_scale_changed(self, scale: float) -> None:
        if self.input_image is None:
            return
        h0, w0 = self.input_image.shape[:2]
        new_w = max(1, round(w0 * scale))
        new_h = max(1, round(h0 * scale))
        self.spn_w.blockSignals(True)
        self.spn_h.blockSignals(True)
        self.spn_w.setValue(new_w)
        self.spn_h.setValue(new_h)
        self.spn_w.blockSignals(False)
        self.spn_h.blockSignals(False)
        self._refresh_preview()

    def _apply_scale_to_size(self) -> None:
        # 显式「应用倍率」按钮:用当前 dsp_scale 重算一次
        self._on_scale_changed(self.dsp_scale.value())
        if self._buf is not None and self.input_image is not None:
            out = self._current_output()
            if out is not None:
                self._buf.push(out, source_tab="尺寸缩放")
        self._on_scale_changed(self.dsp_scale.value())

    # ------------------------------------------------------------------
    # 渲染与导出
    # ------------------------------------------------------------------
    def _current_output(self) -> np.ndarray | None:
        if self.input_image is None:
            return None
        target_w = self.spn_w.value()
        target_h = self.spn_h.value()
        if target_w <= 0 or target_h <= 0:
            return None
        algo = SCALE_ALGORITHMS.get(self.cmb_algo.currentText(), cv2.INTER_NEAREST)
        return cv2.resize(self.input_image, (target_w, target_h), interpolation=algo)

    def _refresh_preview(self) -> None:
        out = self._current_output()
        if out is None:
            return
        self.view_output.set_image(out)
        h, w = out.shape[:2]
        algo = self.cmb_algo.currentText()
        self.status_message(f"已缩放至 {w} × {h}  |  {algo}")

    def _on_save(self, fmt: str) -> None:
        out = self._current_output()
        if out is None:
            return
        ext = "png" if fmt == "png" else "jpg"
        filt = "PNG (*.png)" if fmt == "png" else "JPEG (*.jpg *.jpeg)"
        default_name = f"scaled_{self.spn_w.value()}x{self.spn_h.value()}.{ext}"
        path, _ = QFileDialog.getSaveFileName(
            self, f"导出为 {ext.upper()}", default_name, filt,
        )
        if not path:
            return
        try:
            pil = Image.fromarray(out)
            if fmt == "jpg":
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                pil.save(path, "JPEG", quality=95)
            else:
                pil.save(path, "PNG")
            self.last_saved_path = path
            self.status_message(f"已导出 {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))

    def load_from_buffer(self, image: np.ndarray) -> None:
        """从暂存区双击接收一张图片，作为新的输入原图。"""
        self.input_image = image
        h, w = self.input_image.shape[:2]
        self.lbl_file.setText("(暂存区)")
        self.view_input.set_image(image)
        self.spn_w.blockSignals(True)
        self.spn_h.blockSignals(True)
        self.spn_w.setValue(w)
        self.spn_h.setValue(h)
        self.spn_w.blockSignals(False)
        self.spn_h.blockSignals(False)
        self.dsp_scale.setValue(1.0)
        self._refresh_preview()
        self.btn_save_png.setEnabled(True)
        self.btn_save_jpg.setEnabled(True)
        self.status_message(f"已从暂存区加载 ({w} × {h})")

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
# 主窗口
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Perfect Pixel Tool")
        self.resize(1280, 800)

        # ------- 中央:左侧 Tab 区域 + 右侧暂存区 -------
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)

        # ------- 第一个工具 -------
        self.pixel_tab = PixelRefineWidget()
        self.tabs.addTab(self.pixel_tab, "🎨 像素细化")

        # ------- 第二个工具 -------
        self.scale_tab = ScaleWidget()
        self.tabs.addTab(self.scale_tab, "📐 尺寸缩放")
        ScaleWidget.set_buffer_ref(image_buffer())

        # ------- 第三个工具:去水印 -------
        try:
            from watermark_remover.widget import WatermarkWidget, set_buffer_ref
            self.watermark_tab = WatermarkWidget()
            self.tabs.addTab(self.watermark_tab, "🪄 去水印")
            # 把 buffer 注入 watermark_remover，以便它能 push 图片
            set_buffer_ref(image_buffer())
        except Exception as exc:  # noqa: BLE001
            placeholder_wm = QWidget()
            ph_layout = QVBoxLayout(placeholder_wm)
            ph_layout.setAlignment(Qt.AlignCenter)
            ph_label = QLabel(
                f"⚠️ 去水印模块加载失败\n\n{exc}\n\n请检查 torch / opencv-python 是否已安装。"
            )
            ph_label.setAlignment(Qt.AlignCenter)
            ph_label.setStyleSheet("color: #c33; font-size: 14px;")
            ph_label.setWordWrap(True)
            ph_layout.addWidget(ph_label)
            self.tabs.addTab(placeholder_wm, "🪄 去水印")

        # ------- 预留 Tab -------
        placeholder = QWidget()
        ph_layout = QVBoxLayout(placeholder)
        ph_layout.setAlignment(Qt.AlignCenter)
        ph_label = QLabel(
            "🚧 工具开发中…\n\n下一个工具会在此添加。"
        )
        ph_label.setAlignment(Qt.AlignCenter)
        ph_label.setStyleSheet("color: #888; font-size: 16px;")
        ph_layout.addWidget(ph_label)
        self.tabs.addTab(placeholder, "➕ 即将到来")

        root.addWidget(self.tabs, 1)

        # ------- 右侧暂存区 -------
        self.tray = ImageTrayWidget(image_buffer())
        self.tray.load_request.connect(self._on_tray_load_request)
        root.addWidget(self.tray)

        self.setCentralWidget(central)

        # ------- 工具栏 -------
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        action_open = QAction("打开", self)
        action_open.setShortcut(QKeySequence.Open)
        action_open.triggered.connect(self._dispatch_open)
        toolbar.addAction(action_open)

        action_save = QAction("保存", self)
        action_save.setShortcut(QKeySequence.Save)
        action_save.triggered.connect(self._dispatch_save)
        toolbar.addAction(action_save)

        # ------- 状态栏 -------
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("就绪 — 拖拽图片到窗口,或点击「打开图片」")

        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

    def register_tab(self, widget: QWidget, title: str) -> None:
        self.tabs.insertTab(self.tabs.count() - 1, widget, title)

    # ------------------------------------------------------------------
    # 暂存区双击 → 把图片送入当前 Tab
    # ------------------------------------------------------------------
    def _on_tray_load_request(self, item_id: str) -> None:
        img = image_buffer().get_by_id(item_id)
        if img is None:
            return
        w = self.tabs.currentWidget()
        if w is not None and hasattr(w, "load_from_buffer"):
            w.load_from_buffer(img)

    # ------------------------------------------------------------------
    # 工具栏快捷键:按当前 Tab 分发
    # ------------------------------------------------------------------
    def _current_widget(self) -> QWidget | None:
        return self.tabs.currentWidget()

    def _dispatch_open(self) -> None:
        w = self._current_widget()
        if w is not None and hasattr(w, "on_open"):
            w.on_open()

    def _dispatch_save(self) -> None:
        w = self._current_widget()
        if w is None:
            return
        if hasattr(w, "on_save"):
            w.on_save()
        elif hasattr(w, "_on_save"):
            w._on_save("png")


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
