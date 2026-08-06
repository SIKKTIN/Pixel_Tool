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
from PySide6.QtGui import QAction, QIcon, QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
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
    QButtonGroup,
    QLayout,
    QAbstractButton,
    QStackedLayout,
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
                w.save_requested.connect(self._on_thumb_save)
                w.delete_requested.connect(self._on_thumb_delete)
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

    def _on_thumb_save(self, item_id: str) -> None:
        arr = self._buf.get_by_id(item_id)
        if arr is None:
            return
        # 找到来源信息作为默认文件名
        source_file = ""
        for it in self._buf.items():
            if it["id"] == item_id:
                source_file = it.get("source_file", "") or it.get("source_tab", "")
                break
        default_name = (Path(source_file).stem if source_file else "image") + ".png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            default_name,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            if arr.ndim == 2:
                img = Image.fromarray(arr)
            else:
                img = Image.fromarray(arr[:, :, :3] if arr.shape[2] == 4 else arr)
            ext = Path(path).suffix.lower()
            if ext in (".jpg", ".jpeg"):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(path, "JPEG", quality=95)
            elif ext == ".png":
                if arr.ndim == 3 and arr.shape[2] == 4:
                    img.save(path, "PNG")
                else:
                    img.save(path, "PNG")
            else:
                img.save(path)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def _on_thumb_delete(self, item_id: str) -> None:
        self._buf.remove(item_id)

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
    save_requested = Signal(str)
    delete_requested = Signal(str)

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
        self._lbl_size.setStyleSheet("font-size: 10px; color: #666; background: transparent; border: none;")
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

        if arr.shape[2] == 4:
            qimg = QImage(img_small.tobytes(), tw, th, 4 * tw, QImage.Format_RGBA8888).copy()
        else:
            qimg = QImage(img_small.tobytes(), tw, th, 3 * tw, QImage.Format_RGB888).copy()

        pix = QPixmap.fromImage(qimg)
        self._lbl_img.setPixmap(pix)

    def _update_border(self) -> None:
        if self._is_active:
            self.setStyleSheet(
                "border: 2px solid #4caf50; border-radius: 6px; "
                "background: #e8f5e9;"
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

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        act_save = menu.addAction("保存…")
        act_save.triggered.connect(lambda: self.save_requested.emit(self._item_id))
        menu.addSeparator()
        act_delete = menu.addAction("删除")
        act_delete.triggered.connect(lambda: self.delete_requested.emit(self._item_id))
        menu.exec(event.globalPos())


# 把 src/ 加入 sys.path，使 watermark_remover 子包和 perfect_pixel 可直接 import
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from perfect_pixel import get_perfect_pixel
from perfect_pixel.background_remover import (
    remove_background_color,
    remove_background_channel,
    remove_background_ai,
    remove_background,
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """H x W x 3/4 uint8 -> QPixmap。RGBA 时自动画棋盘格底再合成。"""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)

    h, w = arr.shape[:2]
    if arr.shape[2] == 4:
        # 棋盘格背景 16x16，格灰/白 = 204/232
        chk = np.zeros((h, w, 3), dtype=np.uint8)
        tile = 16
        for row in range(h):
            for col in range(w):
                chk[row, col] = [204, 232][((row // tile) + (col // tile)) % 2]
        # 合成：前景用 alpha 加权叠加
        a = arr[:, :, 3:4].astype(np.float32) / 255.0
        chk = chk.astype(np.float32)
        rgb = arr[:, :, :3].astype(np.float32)
        blended = (rgb * a + chk * (1 - a)).astype(np.uint8)
        # Format_RGB888: QImage 字节顺序 = R-G-B，与 numpy RGB 完全一致
        qimg = QImage(blended.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    else:
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
            img = Image.open(path).convert("RGBA")
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
        self.input_image = np.ascontiguousarray(image, dtype=np.uint8)
        self.view_input.set_image(self.input_image)
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
            Qt.FastTransformation,
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
            img = Image.open(path).convert("RGBA")
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
        self.input_image = np.ascontiguousarray(image, dtype=np.uint8)
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
# 背景去除工具页（新）
# ---------------------------------------------------------------------------

class BGRWorker(QThread):
    """后台背景处理线程，避免大图时 UI 卡死。"""
    finished = Signal(object)   # np.ndarray | None
    failed = Signal(str)

    def __init__(
        self, mode: str,
        input_rgb: np.ndarray,
        bg_color_preview, bg_color_picked,
        bg_color,
        c_tolerance, c_contiguous, c_feather,
        c_anti_alias, c_chroma_key, c_edge_shrink,
        ch_channel, ch_min, ch_max, ch_invert,
        ch_feather, ch_edge_shrink,
        ai_model, ai_edge_shrink,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.input_rgb = input_rgb
        self.bg_color_preview = bg_color_preview
        self.bg_color_picked = bg_color_picked
        self.bg_color = bg_color
        self.c_tolerance = c_tolerance
        self.c_contiguous = c_contiguous
        self.c_feather = c_feather
        self.c_anti_alias = c_anti_alias
        self.c_chroma_key = c_chroma_key
        self.c_edge_shrink = c_edge_shrink
        self.ch_channel = ch_channel
        self.ch_min = ch_min
        self.ch_max = ch_max
        self.ch_invert = ch_invert
        self.ch_feather = ch_feather
        self.ch_edge_shrink = ch_edge_shrink
        self.ai_model = ai_model
        self.ai_edge_shrink = ai_edge_shrink

    def run(self) -> None:
        try:
            result = self._compute()
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _compute(self) -> "np.ndarray | None":
        img = self.input_rgb
        if self.mode == "color":
            bg: Optional[Tuple[int, int, int]] = None
            if self.bg_color_picked and self.bg_color_preview is not None:
                r = int(self.bg_color_preview[2])
                g = int(self.bg_color_preview[1])
                b = int(self.bg_color_preview[0])
                bg = (b, g, r)
            elif self.bg_color is not None:
                r = int(self.bg_color[2])
                g = int(self.bg_color[1])
                b = int(self.bg_color[0])
                bg = (b, g, r)

            return remove_background_color(
                img,
                tolerance=float(self.c_tolerance),
                contiguous_only=self.c_contiguous,
                target_color=bg,
                feather=float(self.c_feather),
                anti_alias=self.c_anti_alias,
                chroma_key=self.c_chroma_key,
                edge_shrink=float(self.c_edge_shrink),
            )
        elif self.mode == "channel":
            return remove_background_channel(
                img,
                channel=self.ch_channel,
                min_threshold=float(self.ch_min),
                max_threshold=float(self.ch_max),
                invert=self.ch_invert,
                feather=float(self.ch_feather),
                edge_shrink=float(self.ch_edge_shrink),
            )
        elif self.mode == "ai":
            return remove_background_ai(
                img,
                model_path=self.ai_model,
                edge_shrink=float(self.ai_edge_shrink),
            )
        return None


class BackgroundRemoverWidget(QWidget):
    """背景去除工具: 三种模式（颜色 / 通道 / AI），支持洪水填充、Chroma Key、边缘收缩、羽化等。"""

    _buf: "ImageBuffer | None" = None

    @staticmethod
    def set_buffer_ref(buf: "ImageBuffer | None") -> None:
        BackgroundRemoverWidget._buf = buf

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buf: "ImageBuffer | None" = BackgroundRemoverWidget._buf

        self.input_image: np.ndarray | None = None   # 原始 RGBA uint8
        self.input_rgb: np.ndarray | None = None      # RGB 用于计算
        self.bg_color: np.ndarray | None = None       # BGR uint8 检测出的背景色
        self.output_rgba: np.ndarray | None = None  # 处理结果 RGBA
        self.last_saved_path: str | None = None
        self._histogram_data: np.ndarray | None = None  # 通道直方图缓存
        self._pending_refresh = False  # 参数已改，等待用户点按钮
        self._dirty_since_load = False  # 加载后是否改过参数
        self._worker: "BGRWorker | None" = None  # 后台处理线程

        # ---- 拖拽 ----
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- 顶部控件栏 --------------------------------------------------
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.btn_open = QPushButton("打开图片…")
        self.btn_open.clicked.connect(self._on_open)
        ctrl_row.addWidget(self.btn_open)

        self.lbl_file = QLabel("未选择文件")
        self.lbl_file.setStyleSheet("color: #666;")
        ctrl_row.addWidget(self.lbl_file, 1)

        ctrl_row.addStretch(1)
        root.addLayout(ctrl_row)

        # ---- 模式切换标签 ------------------------------------------------
        mode_bar = QHBoxLayout()
        mode_bar.setSpacing(6)
        mode_bar.addWidget(QLabel("模式:"))

        self._mode_group = QButtonGroup()
        self._mode_buttons: dict[str, QPushButton] = {}
        for label, val in [("按颜色", "color"), ("按通道", "channel"), ("AI 智能", "ai")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedWidth(80)
            btn.setCursor(Qt.PointingHandCursor)
            self._mode_group.addButton(btn)
            self._mode_buttons[val] = btn
            mode_bar.addWidget(btn)
        mode_bar.addStretch(1)
        root.addLayout(mode_bar)
        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        # 默认选中"按颜色"
        self._mode_buttons["color"].setChecked(True)

        # ---- 操作区 --------------------------------------------------
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_clear_panel = QPushButton("清除")
        self.btn_clear_panel.clicked.connect(self._on_clear_panel)
        self.btn_clear_panel.setEnabled(False)
        action_row.addWidget(self.btn_clear_panel)

        self.btn_export_png = QPushButton("导出 PNG")
        self.btn_export_png.clicked.connect(self._export_png)
        self.btn_export_png.setEnabled(False)
        action_row.addWidget(self.btn_export_png)

        self.btn_export_rgb = QPushButton("导出 RGB")
        self.btn_export_rgb.clicked.connect(self._export_rgb)
        self.btn_export_rgb.setEnabled(False)
        action_row.addWidget(self.btn_export_rgb)

        self.btn_add_to_tray = QPushButton("加入暂存区")
        self.btn_add_to_tray.clicked.connect(self._push_to_buffer)
        self.btn_add_to_tray.setEnabled(False)
        action_row.addWidget(self.btn_add_to_tray)

        action_row.addStretch(1)

        self.btn_detect = QPushButton("重新采样背景色")
        self.btn_detect.clicked.connect(self._detect_bg)
        self.btn_detect.setEnabled(False)
        action_row.addWidget(self.btn_detect)

        self.btn_process = QPushButton("启动处理")
        self.btn_process.clicked.connect(self._do_process)
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet(
            "QPushButton { font-weight: bold; }"
        )
        action_row.addWidget(self.btn_process)

        root.addLayout(action_row)

        # ---- 左右分栏 ----------------------------------------------------
        body = QHBoxLayout()
        body.setSpacing(10)

        # --- 左:参数面板 ---
        self._param_box = QGroupBox("按颜色 — 处理参数")
        self._param_stack = QStackedLayout(self._param_box)
        self._param_stack.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self._param_box, 0)

        # 创建三个模式页面
        self._page_color = QWidget()
        self._page_channel = QWidget()
        self._page_ai = QWidget()
        self._param_stack.addWidget(self._page_color)
        self._param_stack.addWidget(self._page_channel)
        self._param_stack.addWidget(self._page_ai)

        # --- 右:预览 ---
        preview_wrap = QVBoxLayout()
        preview_wrap.setSpacing(10)
        self.view_input = ImageView("原图")
        self.view_output = ImageView("处理结果预览")
        preview_wrap.addWidget(self.view_input, 1)
        preview_wrap.addWidget(self.view_output, 1)
        pw = QWidget()
        pw.setLayout(preview_wrap)
        body.addWidget(pw, 1)
        root.addLayout(body, 1)

        # ---- 构建各模式参数控件 ----
        self._build_color_controls()
        self._build_channel_controls()
        self._build_ai_controls()

    # ------------------------------------------------------------------
    # 控件构建 — 按颜色
    # ------------------------------------------------------------------
    def _build_color_controls(self) -> None:
        self._c_tolerance = QSpinBox()
        self._c_tolerance.setRange(1, 100)
        self._c_tolerance.setValue(10)
        self._c_tolerance.setFixedWidth(100)
        self._c_tolerance.valueChanged.connect(self._mark_dirty)

        self._c_feather = QSpinBox()
        self._c_feather.setRange(0, 20)
        self._c_feather.setValue(0)
        self._c_feather.setFixedWidth(100)
        self._c_feather.valueChanged.connect(self._mark_dirty)

        self._c_edge_shrink = QSpinBox()
        self._c_edge_shrink.setRange(0, 20)
        self._c_edge_shrink.setValue(0)
        self._c_edge_shrink.setFixedWidth(100)
        self._c_edge_shrink.valueChanged.connect(self._mark_dirty)

        self._c_anti_alias = QCheckBox("边缘抗锯齿")
        self._c_anti_alias.setChecked(True)
        self._c_anti_alias.toggled.connect(self._mark_dirty)

        self._c_chroma_key = QCheckBox("Chroma Key 净化（绿幕/蓝幕）")
        self._c_chroma_key.setChecked(True)
        self._c_chroma_key.toggled.connect(self._mark_dirty)

        self._c_contiguous = QCheckBox("仅处理连续像素（四角洪水）")
        self._c_contiguous.setChecked(True)
        self._c_contiguous.toggled.connect(self._mark_dirty)

        self._bg_color_label = QLabel()
        self._bg_color_label.setFixedSize(60, 24)
        self._bg_color_label.setStyleSheet("background: #ffffff; border: 1px solid #888;")
        self._bg_color_preview = np.array([255, 255, 255], dtype=np.uint8)
        self._bg_color_picked = False

        self._bg_color_btn = QPushButton("拾取背景色")
        self._bg_color_btn.setFixedWidth(120)
        self._bg_color_btn.clicked.connect(self._pick_bg_color)
        hl = QHBoxLayout()
        hl.addWidget(self._bg_color_label)
        hl.addWidget(self._bg_color_btn)
        hl.addStretch(1)

        presets_hl = QHBoxLayout()
        for label, hex_val in [("白", "#ffffff"), ("黑", "#000000")]:
            btn = QPushButton(label)
            btn.setFixedWidth(50)
            btn.clicked.connect(
                lambda checked, h=hex_val: self._set_bg_preset(h)
            )
            presets_hl.addWidget(btn)

        lay = QFormLayout(self._page_color)
        lay.setContentsMargins(10, 14, 10, 10)
        lay.setSpacing(8)
        lay.addRow("颜色容差 (%)", self._c_tolerance)
        lay.addRow("边缘羽化 (px)", self._c_feather)
        lay.addRow("边缘收缩 (px)", self._c_edge_shrink)
        lay.addRow("边缘抗锯齿", self._c_anti_alias)
        lay.addRow("Chroma Key 净化", self._c_chroma_key)
        lay.addRow("仅处理连续像素", self._c_contiguous)
        lay.addRow("背景色", hl)
        lay.addRow("预设", presets_hl)

    # ------------------------------------------------------------------
    # 控件构建 — 按通道
    # ------------------------------------------------------------------
    def _build_channel_controls(self) -> None:
        self._ch_channel = QComboBox()
        self._ch_channel.addItems([
            "亮度 (Luminance)", "饱和度 (Saturation)",
            "红色 (Red)", "绿色 (Green)", "蓝色 (Blue)"
        ])
        self._ch_channel.setFixedWidth(180)
        self._ch_channel.currentIndexChanged.connect(self._on_channel_changed)

        self._ch_min = QSpinBox()
        self._ch_min.setRange(0, 255)
        self._ch_min.setValue(0)
        self._ch_min.setFixedWidth(80)
        self._ch_min.valueChanged.connect(self._mark_dirty)

        self._ch_max = QSpinBox()
        self._ch_max.setRange(0, 255)
        self._ch_max.setValue(255)
        self._ch_max.setFixedWidth(80)
        self._ch_max.valueChanged.connect(self._mark_dirty)

        self._ch_invert = QCheckBox("反转遮罩")
        self._ch_invert.toggled.connect(self._mark_dirty)

        self._ch_feather = QSpinBox()
        self._ch_feather.setRange(0, 20)
        self._ch_feather.setValue(0)
        self._ch_feather.setFixedWidth(100)
        self._ch_feather.valueChanged.connect(self._mark_dirty)

        self._ch_edge_shrink = QSpinBox()
        self._ch_edge_shrink.setRange(0, 20)
        self._ch_edge_shrink.setValue(0)
        self._ch_edge_shrink.setFixedWidth(100)
        self._ch_edge_shrink.valueChanged.connect(self._mark_dirty)

        self._hist_canvas = QLabel()
        self._hist_canvas.setFixedHeight(120)
        self._hist_canvas.setMinimumWidth(300)
        self._hist_canvas.setStyleSheet("background: #18181b; border: 1px solid #444;")
        self._hist_canvas.setAlignment(Qt.AlignCenter)
        self._hist_canvas.setText("加载图片后显示直方图")
        self._hist_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        thr_hl = QHBoxLayout()
        thr_hl.addWidget(QLabel("最小:"))
        thr_hl.addWidget(self._ch_min)
        thr_hl.addSpacing(10)
        thr_hl.addWidget(QLabel("最大:"))
        thr_hl.addWidget(self._ch_max)
        thr_hl.addStretch(1)

        lay = QFormLayout(self._page_channel)
        lay.setContentsMargins(10, 14, 10, 10)
        lay.setSpacing(8)
        lay.addRow("通道来源", self._ch_channel)
        lay.addRow("阈值范围", thr_hl)
        lay.addRow("反转遮罩", self._ch_invert)
        lay.addRow("边缘羽化 (px)", self._ch_feather)
        lay.addRow("边缘收缩 (px)", self._ch_edge_shrink)
        lay.addRow("通道直方图", self._hist_canvas)

    # ------------------------------------------------------------------
    # 控件构建 — AI
    # ------------------------------------------------------------------
    def _build_ai_controls(self) -> None:
        # 模式选择: 内置模型 / 自定义路径
        self._ai_mode = QComboBox()
        self._ai_mode.addItems(["内置模型", "自定义路径"])
        self._ai_mode.setFixedWidth(180)
        self._ai_mode.currentIndexChanged.connect(self._on_ai_mode_changed)

        # 内置模型路径（打包后指向 _internal/assets/models/isnet.onnx）
        import sys
        if getattr(sys, "frozen", False):
            base = Path(sys._MEIPASS)
        else:
            base = Path(__file__).parent
        self._ai_builtin_path = str(base / "assets" / "models" / "isnet.onnx")
        self._ai_builtin_label = QLabel()
        self._ai_builtin_label.setStyleSheet("color: #888; font-size: 12px;")
        self._update_builtin_label()

        # 自定义路径控件
        self._ai_path = QLineEdit()
        self._ai_path.setPlaceholderText("选择 ONNX 模型文件路径…")
        self._ai_path.setReadOnly(True)

        btn_browse = QPushButton("浏览…")
        btn_browse.setFixedWidth(70)
        btn_browse.clicked.connect(self._browse_ai_model)

        path_row = QHBoxLayout()
        path_row.addWidget(self._ai_path)
        path_row.addWidget(btn_browse)

        self._ai_edge_shrink = QSpinBox()
        self._ai_edge_shrink.setRange(0, 20)
        self._ai_edge_shrink.setValue(0)
        self._ai_edge_shrink.setFixedWidth(100)

        self._ai_status = QLabel()
        self._ai_status.setStyleSheet("color: #888; font-size: 12px;")

        lay = QFormLayout(self._page_ai)
        lay.setContentsMargins(10, 14, 10, 10)
        lay.setSpacing(8)
        lay.addRow("选择模式", self._ai_mode)
        lay.addRow("内置路径", self._ai_builtin_label)
        lay.addRow("自定义路径", path_row)
        lay.addRow("边缘收缩 (px)", self._ai_edge_shrink)
        lay.addRow("说明", self._ai_status)
        self._on_ai_mode_changed(0)

    def _update_builtin_label(self) -> None:
        exists = Path(self._ai_builtin_path).exists()
        if exists:
            self._ai_builtin_label.setText(f"  {self._ai_builtin_path}")
        else:
            self._ai_builtin_label.setText(f"  [未找到] {self._ai_builtin_path}")
            self._ai_builtin_label.setStyleSheet("color: #c55; font-size: 12px;")

    def _on_ai_mode_changed(self, index: int) -> None:
        if index == 0:  # 内置模型
            self._ai_path.setEnabled(False)
            self._ai_status.setText("使用打包时内置的 isnet 模型，无需额外下载。")
        else:  # 自定义路径
            self._ai_path.setEnabled(True)
            self._ai_status.setText("提示: 推荐使用 isnet-general-use.onnx")

    def _browse_ai_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 ONNX 模型文件", "", "ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if path:
            self._ai_path.setText(path)

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------
    def _on_mode_changed(self, btn: QAbstractButton) -> None:
        for mode_val, b in self._mode_buttons.items():
            if b is btn:
                self._show_mode_controls(mode_val)
                return

    def _show_mode_controls(self, mode: str) -> None:
        index_map = {"color": 0, "channel": 1, "ai": 2}
        self._param_stack.setCurrentIndex(index_map.get(mode, 0))
        title_map = {
            "color": "按颜色 — 处理参数",
            "channel": "按通道 — 处理参数",
            "ai": "AI 智能 — 处理参数",
        }
        self._param_box.setTitle(title_map.get(mode, "参数"))

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
        if local.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            self._load_path(local)

    # ------------------------------------------------------------------
    # 打开 / 加载
    # ------------------------------------------------------------------
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            img = Image.open(path).convert("RGBA")
            self.input_image = np.array(img, dtype=np.uint8)
            self.input_rgb = self.input_image[..., :3]
            h, w = self.input_rgb.shape[:2]
            name = os.path.basename(path)
            self.lbl_file.setText(name)
            self._detect_bg()
            self.btn_detect.setEnabled(True)
            self.btn_clear_panel.setEnabled(True)
            self.btn_export_png.setEnabled(True)
            self.btn_export_rgb.setEnabled(True)
            self.btn_add_to_tray.setEnabled(True)
            self.view_input.set_image(self.input_image)
            self.btn_process.setEnabled(True)
            self.status_message(f"已加载 {name} ({w} × {h})，点击「启动处理」开始去背景")
        except Exception as exc:
            QMessageBox.warning(self, "加载失败", str(exc))

    def _detect_bg(self) -> None:
        """自动采样四角 + 中心区域的平均颜色作为背景色。"""
        if self.input_rgb is None:
            return
        h, w = self.input_rgb.shape[:2]
        m = 8
        ps = 20
        corners = [
            self.input_rgb[m:m + ps, m:m + ps],
            self.input_rgb[m:m + ps, w - m - ps:w - m],
            self.input_rgb[h - m - ps:h - m, m:m + ps],
            self.input_rgb[h - m - ps:h - m, w - m - ps:w - m],
        ]
        all_pixels = np.concatenate([p.reshape(-1, 3) for p in corners], axis=0)
        self.bg_color = all_pixels.mean(axis=0).astype(np.uint8)
        self._update_bg_color_ui()

    def _update_bg_color_ui(self) -> None:
        if self.bg_color is None:
            return
        b, g, r = self.bg_color
        self._bg_color_preview = np.array([b, g, r], dtype=np.uint8)
        self._bg_color_label.setStyleSheet(
            f"background: rgb({r},{g},{b}); border: 1px solid #888;"
        )

    def _set_bg_preset(self, hex_val: str | None) -> None:
        if hex_val is None:
            self._bg_color_preview = None
            self._bg_color_label.setStyleSheet(
                "background: repeating-conic-gradient(#fff 0% 25%, #ddd 0% 50%) 50 / 16px 16px; border: 1px solid #888;"
            )
        else:
            r = int(hex_val[1:3], 16)
            g = int(hex_val[3:5], 16)
            b = int(hex_val[5:7], 16)
            self._bg_color_preview = np.array([b, g, r], dtype=np.uint8)
            self._bg_color_label.setStyleSheet(
                f"background: {hex_val}; border: 1px solid #888;"
            )
        self._bg_color_picked = False
        self._mark_dirty()

    def _pick_bg_color(self) -> None:
        """弹出颜色对话框手动指定背景色。"""
        if self.input_image is None:
            return
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self._bg_color_preview = np.array(
                [color.blue(), color.green(), color.red()], dtype=np.uint8
            )
            self.bg_color = self._bg_color_preview.copy()
            self._bg_color_picked = True
            self._update_bg_color_ui()
            self._mark_dirty()

    # ------------------------------------------------------------------
    # 通道切换（刷新直方图）
    # ------------------------------------------------------------------
    def _on_channel_changed(self) -> None:
        if self.input_rgb is None:
            return
        self._compute_histogram()
        self._draw_histogram()
        self._mark_dirty()

    # ------------------------------------------------------------------
    # 通道直方图计算与绘制
    # ------------------------------------------------------------------
    CHANNEL_MAP = [
        "luminance", "saturation", "red", "green", "blue"
    ]

    def _channel_index_from_combo(self) -> str:
        idx = self._ch_channel.currentIndex()
        return self.CHANNEL_MAP[idx] if 0 <= idx < len(self.CHANNEL_MAP) else "luminance"

    def _compute_histogram(self) -> None:
        """计算通道直方图数据。"""
        if self.input_rgb is None:
            return
        img = self.input_rgb.astype(np.float32)
        r_ch = img[:, :, 0]
        g_ch = img[:, :, 1]
        b_ch = img[:, :, 2]
        channel_str = self._channel_index_from_combo()

        if channel_str == "luminance":
            vals = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
        elif channel_str == "saturation":
            vmax = np.maximum(np.maximum(r_ch, g_ch), b_ch)
            vmin = np.minimum(np.minimum(r_ch, g_ch), b_ch)
            delta = vmax - vmin
            denom = np.where(vmax == 0, 1.0, vmax)
            vals = (delta / denom) * 255.0
        elif channel_str == "red":
            vals = r_ch
        elif channel_str == "green":
            vals = g_ch
        else:
            vals = b_ch

        hist, _ = np.histogram(vals.ravel(), bins=256, range=(0, 256))
        self._histogram_data = hist.astype(np.float32)

    def _draw_histogram(self) -> None:
        """用 QPixmap 绘制通道直方图。"""
        if self._histogram_data is None:
            return

        w = max(self._hist_canvas.width(), 300)
        h = 120
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:, :] = [24, 24, 27]

        hist = self._histogram_data.copy()
        max_val = float(np.percentile(hist[hist > 0], 99)) if hist.max() > 0 else 1.0

        min_thr = self._ch_min.value()
        max_thr = self._ch_max.value()

        for i in range(256):
            bar_h = int(min(h - 8, (hist[i] / max_val) * (h - 8)))
            x = int((i / 255.0) * w)
            in_range = min_thr <= i <= max_thr
            color = [16, 185, 129] if in_range else [113, 113, 122]
            alpha = 165 if in_range else 64
            # blend bar with background
            canvas[h - 4 - bar_h:h - 4, x:x + 1] = np.clip(
                np.array(color, dtype=np.float32) * (alpha / 255.0) +
                24 * ((255 - alpha) / 255.0),
                0, 255
            ).astype(np.uint8)

        # 阈值线
        x_min = int((min_thr / 255.0) * w)
        x_max = int((max_thr / 255.0) * w)
        canvas[:, x_min:x_min + 1, :] = [16, 185, 129]
        canvas[:, x_max:x_max + 1, :] = [16, 185, 129]

        qimg = QImage(canvas.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._hist_canvas.setPixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------
    # 核心: 生成结果
    # ------------------------------------------------------------------
    def _get_current_mode(self) -> str:
        for mode_val, btn in self._mode_buttons.items():
            if btn.isChecked():
                return mode_val
        return "color"

    def _get_result(self) -> Optional[np.ndarray]:
        """根据当前模式和参数生成处理结果。"""
        if self.input_image is None:
            return None
        # 优先使用 RGBA，迭代时 alpha 信息可以保护已透明区域
        if self.input_image.shape[2] == 4:
            img = self.input_image
        else:
            img = self.input_rgb
        mode = self._get_current_mode()

        try:
            if mode == "color":
                bg: Optional[Tuple[int, int, int]] = None
                if self._bg_color_picked and self._bg_color_preview is not None:
                    r, g, b = int(self._bg_color_preview[2]), \
                              int(self._bg_color_preview[1]), \
                              int(self._bg_color_preview[0])
                    bg = (b, g, r)  # RGB -> BGR for the algo
                elif self.bg_color is not None:
                    r, g, b = int(self.bg_color[2]), \
                              int(self.bg_color[1]), \
                              int(self.bg_color[0])
                    bg = (b, g, r)

                return remove_background_color(
                    img,
                    tolerance=float(self._c_tolerance.value()),
                    contiguous_only=self._c_contiguous.isChecked(),
                    target_color=bg,
                    feather=float(self._c_feather.value()),
                    anti_alias=self._c_anti_alias.isChecked(),
                    chroma_key=self._c_chroma_key.isChecked(),
                    edge_shrink=float(self._c_edge_shrink.value()),
                )
            elif mode == "channel":
                return remove_background_channel(
                    img,
                    channel=self.CHANNEL_MAP[self._ch_channel.currentIndex()],
                    min_threshold=float(self._ch_min.value()),
                    max_threshold=float(self._ch_max.value()),
                    invert=self._ch_invert.isChecked(),
                    feather=float(self._ch_feather.value()),
                    edge_shrink=float(self._ch_edge_shrink.value()),
                )
            elif mode == "ai":
                model_path = self._ai_builtin_path if self._ai_mode.currentIndex() == 0 else self._ai_path.text()
                return remove_background_ai(
                    img,
                    model_path=model_path,
                    edge_shrink=float(self._ai_edge_shrink.value()),
                )
        except Exception as exc:
            QMessageBox.warning(self, "处理失败", str(exc))
            return None
        return None

    # ------------------------------------------------------------------
    # 预览刷新
    # ------------------------------------------------------------------
    def _refresh_preview(self) -> None:
        """由 worker 完成信号触发，刷新预览。"""
        pass  # 实际刷新由 _on_worker_done 处理

    def _mark_dirty(self) -> None:
        """标记参数已变动，等待用户点按钮处理。"""
        if self.input_rgb is not None:
            self._pending_refresh = True

    def _do_process(self) -> None:
        """启动后台处理（点按钮 / 切换模式）。"""
        if self.input_rgb is None:
            return

        mode = self._get_current_mode()

        # AI 模式：验证模型文件
        if mode == "ai":
            if self._ai_mode.currentIndex() == 0:
                model_path = self._ai_builtin_path
            else:
                model_path = self._ai_path.text()
            if not model_path or not Path(model_path).is_file():
                QMessageBox.warning(self, "缺少模型", "请选择有效的 ONNX 模型文件（.onnx）")
                return

        # 通道模式：先刷新直方图（同步，快）
        if mode == "channel":
            self._compute_histogram()
            self._draw_histogram()

        # 取消旧的 worker
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(500)
        self._pending_refresh = False
        self.status_message("处理中…")

        self._worker = BGRWorker(
            mode=mode,
            input_rgb=self.input_image if self.input_image is not None and self.input_image.shape[2] == 4 else self.input_rgb,
            bg_color_preview=getattr(self, "_bg_color_preview", None),
            bg_color_picked=getattr(self, "_bg_color_picked", False),
            bg_color=self.bg_color,
            c_tolerance=self._c_tolerance.value(),
            c_contiguous=self._c_contiguous.isChecked(),
            c_feather=self._c_feather.value(),
            c_anti_alias=self._c_anti_alias.isChecked(),
            c_chroma_key=self._c_chroma_key.isChecked(),
            c_edge_shrink=self._c_edge_shrink.value(),
            ch_channel=self.CHANNEL_MAP[self._ch_channel.currentIndex()],
            ch_min=self._ch_min.value(),
            ch_max=self._ch_max.value(),
            ch_invert=self._ch_invert.isChecked(),
            ch_feather=self._ch_feather.value(),
            ch_edge_shrink=self._ch_edge_shrink.value(),
            ai_model=self._ai_builtin_path if self._ai_mode.currentIndex() == 0 else self._ai_path.text(),
            ai_edge_shrink=self._ai_edge_shrink.value(),
        )
        self._worker.finished.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_worker_done(self, result: "np.ndarray | None") -> None:
        """后台处理完成，刷新预览。"""
        if result is None:
            self.status_message("处理未产生结果")
            return
        self.output_rgba = result
        self._display_preview(result)
        fg_ratio = np.mean(result[:, :, 3]) / 255.0 * 100
        mode = self._get_current_mode()
        self.status_message(f"前景比例: {fg_ratio:.1f}%  |  模式: {mode}")

    def _on_worker_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "处理失败", msg)

    def _on_clear_panel(self) -> None:
        """清除当前面板的图片、预览、状态。"""
        self.input_image = None
        self.input_rgb = None
        self.output_rgba = None
        self.bg_color = None
        self.view_input.clear()
        self.view_output.clear()
        self.lbl_file.setText("未加载图片")
        self.btn_detect.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.btn_clear_panel.setEnabled(False)
        self.btn_export_png.setEnabled(False)
        self.btn_export_rgb.setEnabled(False)
        self.btn_add_to_tray.setEnabled(False)
        self._histogram_data = None
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(500)
        self.status_message("已清除")

    def _display_preview(self, rgba: np.ndarray) -> None:
        """棋盘格背景叠加 RGBA 显示。"""
        h, w = rgba.shape[:2]
        cell = 8
        cb = np.zeros((h, w, 3), dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                cb[r, c] = [200, 200, 200] if ((r // cell) + (c // cell)) % 2 == 0 else [120, 120, 120]
        a = rgba[:, :, 3:4].astype(np.float32) / 255.0
        fg = rgba[:, :, :3].astype(np.float32)
        blended = (fg * a + cb.astype(np.float32) * (1.0 - a)).astype(np.uint8)
        self.view_output.set_image(blended)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def _export_png(self) -> None:
        result = self.output_rgba or self._get_result()
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 PNG", self.last_saved_path or "", "PNG (*.png)"
        )
        if not path:
            return
        try:
            Image.fromarray(result, mode="RGBA").save(path, "PNG")
            self.last_saved_path = path
            self.status_message(f"已导出 {path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _export_rgb(self) -> None:
        result = self.output_rgba or self._get_result()
        if result is None:
            return
        bg = self._bg_color_preview
        if bg is None:
            bg = np.array([255, 255, 255], dtype=np.uint8)
        h, w = result.shape[:2]
        b, g, r = bg
        rgb = result[:, :, :3].copy().astype(np.float32)
        a = result[:, :, 3:4].astype(np.float32) / 255.0
        bg_img = np.full((h, w, 3), (b, g, r), dtype=np.uint8).astype(np.float32)
        rgb = (rgb * a + bg_img * (1.0 - a)).astype(np.uint8)

        path, _ = QFileDialog.getSaveFileName(
            self, "导出图片", self.last_saved_path or "",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        try:
            pil_img = Image.fromarray(rgb)
            if path.lower().endswith(".jpg"):
                pil_img.save(path, "JPEG", quality=95)
            else:
                pil_img.save(path, "PNG")
            self.last_saved_path = path
            self.status_message(f"已导出 {path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _push_to_buffer(self) -> None:
        if self._buf is None:
            QMessageBox.warning(self, "暂存区不可用", "暂存区未初始化，请检查主程序")
            return
        result = self.output_rgba if self.output_rgba is not None else self._get_result()
        if result is None:
            QMessageBox.warning(self, "无可用结果", "请先加载图片并启动处理")
            return
        self._buf.push(result, source_tab="去背景")
        self.status_message("已加入暂存区")

    # ------------------------------------------------------------------
    # 从暂存区加载
    # ------------------------------------------------------------------
    def load_from_buffer(self, image: np.ndarray) -> None:
        self.input_image = np.ascontiguousarray(image, dtype=np.uint8)
        self.input_rgb = (
            self.input_image[..., :3]
            if self.input_image.shape[2] == 4
            else self.input_image
        )
        self.lbl_file.setText("(暂存区)")
        self.btn_detect.setEnabled(True)
        self.btn_clear_panel.setEnabled(True)
        self.btn_export_png.setEnabled(True)
        self.btn_export_rgb.setEnabled(True)
        self.btn_add_to_tray.setEnabled(True)
        self.view_input.set_image(self.input_image)
        self.btn_process.setEnabled(True)
        self._detect_bg()
        self.status_message("已从暂存区加载，点击「启动处理」开始去背景")

    # ------------------------------------------------------------------
    # 状态栏
    # ------------------------------------------------------------------
    def status_message(self, msg: str) -> None:
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            sb = win.statusBar()
            if sb is not None:
                sb.showMessage(msg)




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

        # ------- 去背景 Tab -------
        BackgroundRemoverWidget.set_buffer_ref(image_buffer())
        self.bg_tab = BackgroundRemoverWidget()
        self.tabs.addTab(self.bg_tab, "🎭 去背景")

        # ------- 手动编辑 Tab -------
        from manual_editor import ManualEditorWidget
        ManualEditorWidget.set_buffer_ref(image_buffer())
        self.manual_tab = ManualEditorWidget()
        self.tabs.addTab(self.manual_tab, "✏️ 手动编辑")

        # ------- 图像切割 Tab -------
        from image_splitter import ImageSplitterWidget
        ImageSplitterWidget.set_buffer_ref(image_buffer())
        self.splitter_tab = ImageSplitterWidget()
        self.tabs.addTab(self.splitter_tab, "✂️ 图像切割")

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
    # 解析图标路径:开发期读 assets/,打包后读 _MEIPASS/assets/
    base_dir = getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)
    _png = Path(base_dir) / "assets" / "app_icon_src.png"
    _ico = Path(base_dir) / "assets" / "app_icon.ico"

    log_path = Path(base_dir) / "icon_debug.log"
    try:
        with open(log_path, "w") as f:
            f.write(f"base_dir={base_dir}\n")
            f.write(f"_png={_png} exists={_png.exists()}\n")
            f.write(f"_ico={_ico} exists={_ico.exists()}\n")
    except Exception:
        pass

    pix = None
    if _png.exists():
        pix = QPixmap(str(_png))
        if not pix.isNull():
            try:
                with open(log_path, "a") as f:
                    f.write(f"PNG loaded OK: {pix.width()}x{pix.height()}\n")
            except Exception:
                pass
    if pix is None or pix.isNull():
        if _ico.exists():
            pix = QPixmap(str(_ico))
            if not pix.isNull():
                try:
                    with open(log_path, "a") as f:
                        f.write(f"ICO fallback loaded OK: {pix.width()}x{pix.height()}\n")
                except Exception:
                    pass
    if pix is None or pix.isNull():
        try:
            with open(log_path, "a") as f:
                f.write("WARNING: no icon loaded\n")
        except Exception:
            pass
    else:
        icon = QIcon(pix)
        for s in (16, 24, 32, 48, 64, 128, 256):
            icon.addPixmap(pix.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        app.setWindowIcon(icon)

    window = MainWindow()
    window.show()

    # --- Win32 API: 强制刷新窗口图标句柄（解决标题栏/任务栏绿块） ---
    if sys.platform == "win32":
        import ctypes
        try:
            user32 = ctypes.windll.user32
            SendMessageW = user32.SendMessageW

            def set_icon(hwnd, icon_handle, which):
                SendMessageW(hwnd, 0x0080, which, icon_handle)

            hwnd = int(window.winId())
            LoadImageW = user32.LoadImageW
            _ico_path = str(_ico if _ico.exists() else _png)
            hicon = LoadImageW(None, _ico_path, 1, 0, 0, 2 | 0x10)
            if hicon:
                set_icon(hwnd, hicon, 0)
                set_icon(hwnd, hicon, 1)
                with open(log_path, "a") as f:
                    f.write(f"WM_SETICON hicon={hicon}\n")
            else:
                with open(log_path, "a") as f:
                    f.write("LoadImageW returned 0\n")
        except Exception as ex:
            with open(log_path, "a") as f:
                f.write(f"Win32 error: {ex}\n")

    try:
        with open(log_path, "a") as f:
            f.write("setWindowIcon done\n")
    except Exception:
        pass

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
