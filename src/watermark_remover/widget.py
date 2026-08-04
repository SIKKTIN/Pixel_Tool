"""WatermarkWidget —— PerfectPixelTool 的「去水印」工具页。

把 Test 项目(WatermarkRemover) 的 SLBR / LaMa 能力接入为 PySide6 桌面 Tab。
- 自动模式: SLBR (无需蒙版)
- 手动模式: LaMa  (画笔涂抹蒙版)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal, QPoint, QRect
from PySide6.QtGui import (
    QAction,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from . import LaMaModel, SlbrRunner
from .lama_model import LAMA_MODEL_PATH
from .slbr_runner import DEFAULT_TILE_BATCH, DEFAULT_TILE_SIZE

logger = logging.getLogger(__name__)

# 模型目录: 沿用 LaMa 的默认 MODEL_DIR 解析逻辑
_MODEL_DIR = Path(LAMA_MODEL_PATH).parent


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """H x W x 3 uint8 -> QPixmap (RGB)."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def bgr_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        return None
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def load_image_rgb(path: str) -> np.ndarray:
    """支持中文路径的 RGB 读取."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


# ---------------------------------------------------------------------------
# 后台工作线程
# ---------------------------------------------------------------------------

class SlbrWorker(QThread):
    finished_ok = Signal(object, object)   # clean_rgb, mask_rgb
    failed = Signal(str)

    def __init__(self, image_rgb: np.ndarray, tile_size: int, tile_batch: int):
        super().__init__()
        self.image_rgb = image_rgb
        self.tile_size = tile_size
        self.tile_batch = tile_batch

    def run(self) -> None:
        try:
            image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
            runner = SlbrRunner(model_dir=_MODEL_DIR, device="cuda" if _cuda_available() else "cpu")
            if not runner.installed:
                self.failed.emit(f"SLBR 模型未找到: {runner.checkpoint_path}")
                return
            clean_bgr, mask_bgr = runner.infer_bgr(
                image_bgr,
                tile_size=self.tile_size,
                tile_batch=self.tile_batch,
            )
            self.finished_ok.emit(bgr_to_rgb(clean_bgr), bgr_to_rgb(mask_bgr))
        except Exception as exc:
            logger.exception("SLBR inference failed")
            self.failed.emit(str(exc))


class LamaWorker(QThread):
    finished_ok = Signal(object)   # result_rgb
    failed = Signal(str)

    def __init__(self, image_rgb: np.ndarray, mask_gray: np.ndarray):
        super().__init__()
        self.image_rgb = image_rgb
        self.mask_gray = mask_gray

    def run(self) -> None:
        try:
            model = LaMaModel(device="cuda" if _cuda_available() else "cpu")
            result_bgr = model(self.image_rgb, self.mask_gray)
            self.finished_ok.emit(bgr_to_rgb(result_bgr))
        except Exception as exc:
            logger.exception("LaMa inference failed")
            self.failed.emit(str(exc))


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _push_to_buffer(image: np.ndarray, source_tab: str) -> None:
    """把图片推入全局 ImageBuffer（通过 desktop_app.py 注入的引用）。"""
    # 在 desktop_app.py 的 ImageBuffer 初始化后此引用被设置
    if hasattr(_push_to_buffer, "_buffer"):
        _push_to_buffer._buffer.push(image, source_tab=source_tab)


def set_buffer_ref(buffer_obj) -> None:
    _push_to_buffer._buffer = buffer_obj


# ---------------------------------------------------------------------------
# 蒙版绘制画布
# ---------------------------------------------------------------------------

class MaskCanvas(QWidget):
    """蒙版绘制画布 —— 用户在原图上画红色蒙版(LaMa 模式).

    特性:
        - 自适应: 按 widget 尺寸等比缩放原图, 不超出可视范围
        - 滚轮缩放: Ctrl+滚轮放大缩小(画笔大小同步缩放)
        - 拖拽平移: 在画布空白处按住中键 / 滚轮键拖动
        - 缩放还原: 双击回到适配屏幕的初始缩放
    """

    mask_changed = Signal()

    MIN_SCALE = 0.05
    MAX_SCALE = 10.0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._mask_pixmap: Optional[QPixmap] = None  # 原图尺寸的全透明蒙版
        self._drawing = False
        self._panning = False
        self._last_point: Optional[QPoint] = None
        self._brush_size = 30
        self._tool = "brush"  # 'brush' or 'eraser'

        # 视图变换
        self._scale = 1.0
        self._offset_x = 0.0  # 原图坐标 -> widget 坐标: widget_xy = img_xy * scale + offset
        self._offset_y = 0.0

        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 滚轮事件不需焦点
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def set_image(self, image_rgb: np.ndarray):
        self._pixmap = numpy_to_qpixmap(image_rgb)
        h, w, _ = image_rgb.shape
        # 蒙版层(原图尺寸): 全透明
        self._mask_pixmap = QPixmap(w, h)
        self._mask_pixmap.fill(Qt.transparent)

        # 自适应: 等比缩放到 widget 大小(留 4px 边距)
        self._fit_to_widget()
        self.update()

    def has_pixmap(self) -> bool:
        return self._pixmap is not None

    def clear_mask(self):
        if self._mask_pixmap is None:
            return
        self._mask_pixmap.fill(Qt.transparent)
        self.update()
        self.mask_changed.emit()

    def set_brush_size(self, size: int):
        self._brush_size = max(1, int(size))

    def set_tool(self, tool: str):
        if tool in ("brush", "eraser"):
            self._tool = tool
        self.setCursor(Qt.CrossCursor if self._tool == "brush" else Qt.PointingHandCursor)

    def get_mask_gray(self, target_shape) -> np.ndarray:
        """导出对齐到原图尺寸的灰度蒙版 (255=被涂抹区域)."""
        if self._mask_pixmap is None:
            return np.zeros(target_shape[:2], dtype=np.uint8)
        img = self._mask_pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        w, h = img.width(), img.height()
        ptr = img.bits().tobytes()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
        alpha = arr[..., 3]
        if alpha.shape[:2] != target_shape[:2]:
            alpha = cv2.resize(alpha, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        return (alpha > 0).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # 视图变换
    # ------------------------------------------------------------------
    def _fit_to_widget(self):
        """自适应: 把整张图等比缩放到 widget 大小, 居中."""
        if self._pixmap is None:
            return
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        margin = 4
        avail_w = max(1, widget_w - 2 * margin)
        avail_h = max(1, widget_h - 2 * margin)
        sx = avail_w / self._pixmap.width()
        sy = avail_h / self._pixmap.height()
        self._scale = min(sx, sy)
        # 居中
        disp_w = self._pixmap.width() * self._scale
        disp_h = self._pixmap.height() * self._scale
        self._offset_x = (widget_w - disp_w) / 2.0
        self._offset_y = (widget_h - disp_h) / 2.0

    def _widget_to_image(self, pt: QPoint) -> QPoint:
        """widget 坐标 -> 原图坐标."""
        ix = (pt.x() - self._offset_x) / max(self._scale, 1e-6)
        iy = (pt.y() - self._offset_y) / max(self._scale, 1e-6)
        return QPoint(int(round(ix)), int(round(iy)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 第一次 resize 时自适应
        if self._pixmap is not None and self._scale <= 0:
            self._fit_to_widget()

    def wheelEvent(self, event):
        """Ctrl+滚轮缩放, 围绕鼠标位置缩放."""
        if self._pixmap is None:
            return
        if not (event.modifiers() & Qt.ControlModifier):
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, self._scale * factor))

        # 以鼠标点为锚点缩放, 保持鼠标点处的图像坐标不变
        mouse = event.position()
        # 原图坐标 = (widget - offset) / scale
        ix = (mouse.x() - self._offset_x) / self._scale
        iy = (mouse.y() - self._offset_y) / self._scale
        self._scale = new_scale
        self._offset_x = mouse.x() - ix * self._scale
        self._offset_y = mouse.y() - iy * self._scale

        # 画笔大小同步缩放(保持视觉一致)
        self._brush_size = max(1, int(self._brush_size * factor))
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self._pixmap is None:
            return
        self._fit_to_widget()
        self.update()

    # ------------------------------------------------------------------
    # 鼠标事件
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        # 中键 / 右键 -> 平移
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._last_point = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        self._drawing = True
        img_pt = self._widget_to_image(event.position().toPoint())
        # 限制在图片范围内
        img_pt.setX(max(0, min(self._pixmap.width() - 1, img_pt.x())))
        img_pt.setY(max(0, min(self._pixmap.height() - 1, img_pt.y())))
        self._last_point = img_pt
        self._paint_at(img_pt)

    def mouseMoveEvent(self, event):
        if self._panning and self._last_point is not None:
            cur = event.position().toPoint()
            dx = cur.x() - self._last_point.x()
            dy = cur.y() - self._last_point.y()
            self._offset_x += dx
            self._offset_y += dy
            self._last_point = cur
            self.update()
            return
        if not self._drawing or self._last_point is None:
            return
        cur_img = self._widget_to_image(event.position().toPoint())
        cur_img.setX(max(0, min(self._pixmap.width() - 1, cur_img.x())))
        cur_img.setY(max(0, min(self._pixmap.height() - 1, cur_img.y())))
        self._paint_line(self._last_point, cur_img)
        self._last_point = cur_img

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = False
            self._last_point = None
            self.setCursor(Qt.CrossCursor if self._tool == "brush" else Qt.PointingHandCursor)
            return
        if event.button() == Qt.LeftButton:
            self._drawing = False
            self._last_point = None
            self.mask_changed.emit()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def _paint_at(self, pt: QPoint):
        if self._mask_pixmap is None:
            return
        painter = QPainter(self._mask_pixmap)
        # 蒙版层用原图坐标系, 笔刷大小按缩放反转补偿
        visual_brush = max(1, int(self._brush_size / max(self._scale, 1e-6)))
        pen = QPen(QColor(255, 50, 50, 255), visual_brush,
                   Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 50, 50, 255) if self._tool == "brush" else Qt.transparent)
        painter.drawPoint(pt)
        painter.end()
        self.update()

    def _paint_line(self, p1: QPoint, p2: QPoint):
        if self._mask_pixmap is None:
            return
        painter = QPainter(self._mask_pixmap)
        visual_brush = max(1, int(self._brush_size / max(self._scale, 1e-6)))
        if self._tool == "brush":
            pen = QPen(QColor(255, 50, 50, 255), visual_brush,
                       Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(p1, p2)
        else:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            pen = QPen(Qt.transparent, visual_brush,
                       Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(p1, p2)
        painter.end()
        self.update()

    # ------------------------------------------------------------------
    # 渲染: 把原图 + 蒙版按 scale/offset 绘制到 widget
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        if self._pixmap is None:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "(请打开图片)")
            return

        # 1) 原图(按 scale + offset)
        target_size = self._pixmap.size() * self._scale
        scaled = self._pixmap.scaled(
            int(target_size.width()), int(target_size.height()),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        painter.drawPixmap(int(self._offset_x), int(self._offset_y), scaled)

        # 2) 蒙版
        if self._mask_pixmap is not None:
            scaled_mask = self._mask_pixmap.scaled(
                int(target_size.width()), int(target_size.height()),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            painter.setOpacity(0.6)
            painter.drawPixmap(int(self._offset_x), int(self._offset_y), scaled_mask)
            painter.setOpacity(1.0)

        # 3) 调试信息
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(8, 18, f"缩放 {self._scale * 100:.0f}%  |  Ctrl+滚轮缩放  |  双击重置")


# ---------------------------------------------------------------------------
# 预览面板 (原图 / 结果 二选一对比)
# ---------------------------------------------------------------------------

class ZoomImageLabel(QLabel):
    """可缩放/平移的图片预览 —— 支持 Ctrl+滚轮缩放、拖拽平移、双击重置."""

    MIN_SCALE = 0.05
    MAX_SCALE = 10.0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._panning = False
        self._last_pos: Optional[QPoint] = None

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #1e1e1e; color: #888;")
        self.setFocusPolicy(Qt.StrongFocus)

    def set_image(self, rgb: np.ndarray):
        self._pixmap = numpy_to_qpixmap(rgb)
        self._fit()
        self.update()

    def clear(self):
        super().clear()
        self._pixmap = None
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self.setText("(无图片)")

    def _fit(self):
        if self._pixmap is None:
            return
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        margin = 4
        sx = (widget_w - 2 * margin) / self._pixmap.width()
        sy = (widget_h - 2 * margin) / self._pixmap.height()
        self._scale = max(self.MIN_SCALE, min(self.MAX_SCALE, min(sx, sy)))
        disp_w = self._pixmap.width() * self._scale
        disp_h = self._pixmap.height() * self._scale
        self._offset_x = (widget_w - disp_w) / 2.0
        self._offset_y = (widget_h - disp_h) / 2.0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap is not None and self._scale <= 0:
            self._fit()

    def wheelEvent(self, event):
        if self._pixmap is None:
            return
        if not (event.modifiers() & Qt.ControlModifier):
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, self._scale * factor))
        mouse = event.position()
        ix = (mouse.x() - self._offset_x) / self._scale
        iy = (mouse.y() - self._offset_y) / self._scale
        self._scale = new_scale
        self._offset_x = mouse.x() - ix * self._scale
        self._offset_y = mouse.y() - iy * self._scale
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self._pixmap is None:
            return
        self._fit()
        self.update()

    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._last_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._last_pos is not None:
            cur = event.position().toPoint()
            self._offset_x += cur.x() - self._last_pos.x()
            self._offset_y += cur.y() - self._last_pos.y()
            self._last_pos = cur
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = False
            self._last_pos = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        if self._pixmap is None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        target_w = int(self._pixmap.width() * self._scale)
        target_h = int(self._pixmap.height() * self._scale)
        scaled = self._pixmap.scaled(
            target_w, target_h,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        painter.drawPixmap(int(self._offset_x), int(self._offset_y), scaled)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(8, 18, f"缩放 {self._scale * 100:.0f}%  |  Ctrl+滚轮缩放  |  双击重置")


class ResultPanel(QWidget):
    """结果展示面板 —— 用 QStackedWidget 在「原图」「结果」「蒙版」间切换."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.title = QLabel("(无图片)")
        self.title.setStyleSheet("font-weight: bold;")
        bar.addWidget(self.title)
        bar.addStretch(1)

        self.btn_orig = QPushButton("原图")
        self.btn_orig.setCheckable(True)
        self.btn_orig.setChecked(True)
        self.btn_result = QPushButton("结果")
        self.btn_result.setCheckable(True)
        self.btn_mask = QPushButton("蒙版")
        self.btn_mask.setCheckable(True)
        for b in (self.btn_orig, self.btn_result, self.btn_mask):
            b.setFixedWidth(60)
            bar.addWidget(b)
        v.addLayout(bar)

        self.stack = QStackedWidget()
        self.lbl_orig = ZoomImageLabel()
        self.lbl_result = ZoomImageLabel()
        self.lbl_mask = ZoomImageLabel()
        for w in (self.lbl_orig, self.lbl_result, self.lbl_mask):
            self.stack.addWidget(w)
        v.addWidget(self.stack, 1)

        self._images: dict[str, np.ndarray] = {}
        self.btn_orig.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_result.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_mask.clicked.connect(lambda: self.stack.setCurrentIndex(2))

    def set_original(self, rgb: np.ndarray):
        self._images["orig"] = rgb
        self.lbl_orig.set_image(rgb)
        h, w, _ = rgb.shape
        self.title.setText(f"{w} × {h} px")

    def set_result(self, rgb: np.ndarray):
        self._images["result"] = rgb
        self.lbl_result.set_image(rgb)
        h, w, _ = rgb.shape
        self.title.setText(f"结果 {w} × {h} px")
        self.btn_result.setChecked(True)
        self.stack.setCurrentIndex(1)

    def set_mask(self, rgb: np.ndarray):
        self._images["mask"] = rgb
        self.lbl_mask.set_image(rgb)

    def clear_all(self):
        self._images.clear()
        for lbl in (self.lbl_orig, self.lbl_result, self.lbl_mask):
            lbl.clear()
        self.btn_orig.setChecked(True)
        self.stack.setCurrentIndex(0)
        self.title.setText("(无图片)")


# ---------------------------------------------------------------------------
# 主 Widget
# ---------------------------------------------------------------------------

class WatermarkWidget(QWidget):
    """PerfectPixelTool 的去水印 Tab."""

    MODES = [("SLBR 自动", "slbr"), ("LaMa 手动", "lama")]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.input_image: Optional[np.ndarray] = None
        self.result_image: Optional[np.ndarray] = None
        self.last_saved_path: Optional[str] = None
        self.worker: Optional[QThread] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- 顶部:模式选择 + 主控件 ----
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.btn_open = QPushButton("打开图片…")
        self.btn_open.clicked.connect(self.on_open)
        ctrl_row.addWidget(self.btn_open)

        self.btn_save = QPushButton("保存结果…")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)
        ctrl_row.addWidget(self.btn_save)

        ctrl_row.addSpacing(16)

        ctrl_row.addWidget(QLabel("模式:"))
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for i, (label, key) in enumerate(self.MODES):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            self.mode_group.addButton(btn, i)
            self.mode_buttons[key] = btn
            ctrl_row.addWidget(btn)
        self.mode_group.idClicked.connect(self._on_mode_changed)

        ctrl_row.addStretch(1)
        root.addLayout(ctrl_row)

        # ---- 中部:参数面板 + 主画布/结果 (左右布局) ----
        body = QHBoxLayout()
        body.setSpacing(10)

        # 左:参数 + 蒙版画布(仅 LaMa 模式)
        left_box = QVBoxLayout()
        left_box.setSpacing(8)

        # SLBR 参数
        self.slbr_box = self._build_slbr_params()
        left_box.addWidget(self.slbr_box)

        # LaMa 参数
        self.lama_box = self._build_lama_params()
        left_box.addWidget(self.lama_box)
        self.lama_box.setVisible(False)

        # 蒙版画布 (LaMa 模式)
        self.mask_canvas = MaskCanvas()
        self.mask_canvas.setMinimumHeight(280)
        left_box.addWidget(self.mask_canvas, 1)

        body.addLayout(left_box, 1)

        # 右:结果对比
        self.result_panel = ResultPanel()
        body.addWidget(self.result_panel, 1)

        root.addLayout(body, 1)

        # ---- 底部:运行按钮 + 状态 ----
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self.btn_clear_mask = QPushButton("清除蒙版")
        self.btn_clear_mask.clicked.connect(self._on_clear_mask)
        self.btn_clear_mask.setEnabled(False)
        bottom.addWidget(self.btn_clear_mask)

        bottom.addStretch(1)

        self.btn_run = QPushButton("开始处理")
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_run.setEnabled(False)
        bottom.addWidget(self.btn_run)

        root.addLayout(bottom)

        # 拖拽支持
        self.setAcceptDrops(True)
        self.mask_canvas.setAcceptDrops(False)

    # ------------------------------------------------------------------
    # 参数面板构建
    # ------------------------------------------------------------------
    def _build_slbr_params(self) -> QGroupBox:
        box = QGroupBox("SLBR 参数")
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 14, 10, 10)

        # tile_size
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("分块大小:"))
        self.cmb_tile_size = QComboBox()
        self.cmb_tile_size.addItems(["256", "384", "512"])
        self.cmb_tile_size.setCurrentText(str(DEFAULT_TILE_SIZE))
        row1.addWidget(self.cmb_tile_size)
        row1.addStretch(1)
        v.addLayout(row1)

        # tile_batch
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("批处理大小:"))
        self.sld_batch = QSlider(Qt.Horizontal)
        self.sld_batch.setRange(1, 8)
        self.sld_batch.setValue(DEFAULT_TILE_BATCH)
        row2.addWidget(self.sld_batch)
        self.lbl_batch = QLabel(str(DEFAULT_TILE_BATCH))
        self.lbl_batch.setFixedWidth(28)
        row2.addWidget(self.lbl_batch)
        self.sld_batch.valueChanged.connect(lambda v: self.lbl_batch.setText(str(v)))
        v.addLayout(row2)

        return box

    def _build_lama_params(self) -> QGroupBox:
        box = QGroupBox("LaMa 蒙版工具")
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 14, 10, 10)

        # 工具切换 (brush / eraser)
        row_tool = QHBoxLayout()
        self.btn_brush = QPushButton("画笔")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setChecked(True)
        self.btn_eraser = QPushButton("橡皮擦")
        self.btn_eraser.setCheckable(True)
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_brush, 0)
        self.tool_group.addButton(self.btn_eraser, 1)
        self.tool_group.idClicked.connect(self._on_tool_changed)
        row_tool.addWidget(QLabel("工具:"))
        row_tool.addWidget(self.btn_brush)
        row_tool.addWidget(self.btn_eraser)
        row_tool.addStretch(1)
        v.addLayout(row_tool)

        # 画笔大小
        row_size = QHBoxLayout()
        row_size.addWidget(QLabel("画笔大小:"))
        self.sld_brush = QSlider(Qt.Horizontal)
        self.sld_brush.setRange(5, 100)
        self.sld_brush.setValue(30)
        self.sld_brush.valueChanged.connect(self._on_brush_changed)
        row_size.addWidget(self.sld_brush)
        self.lbl_brush = QLabel("30")
        self.lbl_brush.setFixedWidth(28)
        row_size.addWidget(self.lbl_brush)
        v.addLayout(row_size)

        return box

    # ------------------------------------------------------------------
    # 槽函数
    # ------------------------------------------------------------------
    def _on_mode_changed(self, idx: int):
        is_lama = self.MODES[idx][1] == "lama"
        self.slbr_box.setVisible(not is_lama)
        self.lama_box.setVisible(is_lama)
        self.btn_clear_mask.setEnabled(is_lama and self.mask_canvas.has_pixmap())

    def _on_tool_changed(self, idx: int):
        self.mask_canvas.set_tool("brush" if idx == 0 else "eraser")

    def _on_brush_changed(self, v: int):
        self.mask_canvas.set_brush_size(v)
        self.lbl_brush.setText(str(v))

    def _on_clear_mask(self):
        self.mask_canvas.clear_mask()

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*.*)",
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str):
        try:
            self.input_image = load_image_rgb(path)
        except Exception as exc:
            QMessageBox.critical(self, "打开失败", str(exc))
            return

        # 重置状态
        self.result_image = None
        self.last_saved_path = None
        self.btn_save.setEnabled(False)
        self.btn_run.setEnabled(True)

        # 更新画布
        self.mask_canvas.set_image(self.input_image)
        self.mask_canvas.clear_mask()
        self.result_panel.clear_all()
        self.result_panel.set_original(self.input_image)

        # LaMa 模式下允许清除蒙版按钮
        is_lama = self.mode_buttons["lama"].isChecked()
        self.btn_clear_mask.setEnabled(is_lama)

        self.status_message(f"已加载 {Path(path).name} ({self.input_image.shape[1]} × {self.input_image.shape[0]})")

    def on_run(self):
        if self.input_image is None:
            QMessageBox.information(self, "提示", "请先打开一张图片")
            return
        if self.worker is not None and self.worker.isRunning():
            return

        # 判断当前模式
        is_lama = self.mode_buttons["lama"].isChecked()

        if is_lama:
            mask_gray = self.mask_canvas.get_mask_gray(self.input_image.shape)
            if mask_gray.sum() == 0:
                QMessageBox.information(self, "提示", "请先用画笔涂抹要修复的区域")
                return
            self.btn_run.setEnabled(False)
            self.btn_run.setText("处理中…")
            self.worker = LamaWorker(image_rgb=self.input_image, mask_gray=mask_gray)
            self.worker.finished_ok.connect(self._on_lama_done)
            self.worker.failed.connect(self._on_worker_failed)
            self.worker.start()
            self.status_message("LaMa 推理中…")
        else:
            tile_size = int(self.cmb_tile_size.currentText())
            tile_batch = int(self.sld_batch.value())
            self.btn_run.setEnabled(False)
            self.btn_run.setText("处理中…")
            self.worker = SlbrWorker(
                image_rgb=self.input_image,
                tile_size=tile_size,
                tile_batch=tile_batch,
            )
            self.worker.finished_ok.connect(self._on_slbr_done)
            self.worker.failed.connect(self._on_worker_failed)
            self.worker.start()
            self.status_message("SLBR 推理中…")

    def _on_slbr_done(self, clean_rgb: np.ndarray, mask_rgb: np.ndarray):
        self.result_image = clean_rgb
        self.result_panel.set_result(clean_rgb)
        self.result_panel.set_mask(mask_rgb)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始处理")
        self.btn_save.setEnabled(True)
        # 推入暂存区
        _push_to_buffer(clean_rgb, "去水印(SLBR)")
        self.status_message(
            f"SLBR 完成  |  输出 {clean_rgb.shape[1]} × {clean_rgb.shape[0]} px"
        )

    def _on_lama_done(self, result_rgb: np.ndarray):
        self.result_image = result_rgb
        self.result_panel.set_result(result_rgb)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始处理")
        self.btn_save.setEnabled(True)
        # 推入暂存区
        _push_to_buffer(result_rgb, "去水印(LaMa)")
        self.status_message(
            f"LaMa 完成  |  输出 {result_rgb.shape[1]} × {result_rgb.shape[0]} px"
        )

    def _on_worker_failed(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始处理")
        QMessageBox.warning(self, "处理失败", msg)
        self.status_message(f"处理失败: {msg}")

    def on_save(self):
        if self.result_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "output_clean.png",
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if not path:
            return
        try:
            Image.fromarray(self.result_image).save(path)
            self.last_saved_path = path
            self.status_message(f"已保存到 {path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    # ------------------------------------------------------------------
    # 拖拽
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        local = urls[0].toLocalFile()
        if local:
            self.load_path(local)

    def load_from_buffer(self, image: np.ndarray) -> None:
        """从暂存区双击接收一张图片，作为新的输入。"""
        self.input_image = image
        self.result_image = None
        self.last_saved_path = None
        self.btn_save.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.mask_canvas.set_image(image)
        self.mask_canvas.clear_mask()
        self.result_panel.clear_all()
        self.result_panel.set_original(image)
        h, w = image.shape[:2]
        is_lama = self.mode_buttons["lama"].isChecked()
        self.btn_clear_mask.setEnabled(is_lama)
        self.status_message(f"已从暂存区加载 ({w} × {h})")

    # ------------------------------------------------------------------
    # 状态栏
    # ------------------------------------------------------------------
    def status_message(self, text: str):
        win = self.window()
        if isinstance(win, QMainWindow):
            sb = win.statusBar()
            if sb is not None:
                sb.showMessage(text, 5000)
