"""
手动编辑模块 — 画笔（恢复 / 着色）+ 橡皮擦（擦除 alpha）画在 RGBA 图上。

用法：
    from manual_editor import ManualEditorWidget
    ManualEditorWidget.set_buffer_ref(buf)
    tabs.addTab(ManualEditorWidget(), "✏️ 手动编辑")

支持的操作:
- 打开本地图片 / 从暂存区载入
- 用画笔从源色取色（吸管）或设置前景色，涂抹前景色到 RGBA 的 RGB 通道
- 用橡皮擦把不透明区域擦成透明（alpha=0）
- 撤销/重做（最近 30 步快照）
- 调整笔刷大小、硬度、模式（橡皮擦 / 画笔）
- 加入暂存区 / 导出 PNG
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from PIL import Image
from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, Signal, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QBrush, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# 复用同项目的预览控件 + utils
from desktop_app import ImageView
from perfect_pixel.app_core.image_buffer import ImageBuffer


# ---------------------------------------------------------------------------
# 颜色按钮（点击弹出 QColorDialog）
# ---------------------------------------------------------------------------

class ColorSwatchButton(QPushButton):
    """一个可以显示当前颜色、点击后弹出 QColorDialog 的按钮。"""

    def __init__(self, initial: QColor, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(40, 28)
        self._color = QColor(initial)
        self.clicked.connect(self._pick)
        self._refresh()

    def setColor(self, c: QColor) -> None:
        self._color = QColor(c)
        self._refresh()

    def color(self) -> QColor:
        return QColor(self._color)

    def _pick(self) -> None:
        c = QColorDialog.getColor(self._color, self, "选择颜色", QColorDialog.ShowAlphaChannel)
        if c.isValid():
            self._color = c
            self._refresh()

    def _refresh(self) -> None:
        rgba = (
            f"background-color: rgba({self._color.red()}, {self._color.green()}, {self._color.blue()}, {self._color.alpha()});"
        )
        self.setStyleSheet(
            "QPushButton { border: 1px solid #888; " + rgba + " }"
        )


# ---------------------------------------------------------------------------
# 可编辑画布
# ---------------------------------------------------------------------------

class EditableImageView(ImageView):
    """在 ImageView 基础上支持鼠标绘制。

    维护一张与图像同尺寸的 RGBA 画布（self._layer）。
    笔刷事件触发 paint_stroke -> 修改 _layer 像素 -> 刷新合成预览。
    """

    stroke_started = Signal()    # 鼠标落下，新一笔开始
    stroke_moved   = Signal()    # 拖动中
    stroke_finished = Signal()   # 释放
    cursor_moved   = Signal(int, int)  # 当前鼠标对应的图像坐标（用于状态栏）

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        # --- 编辑画布（与图像同尺寸，ARGB32_Premultiplied 让 QPainter 合成更稳定） ---
        self._canvas: QImage | None = None       # 原图尺寸的 ARGB32 图
        self._bg_pixmap: QPixmap | None = None   # 同尺寸的棋盘格背景（一次生成，永久缓存）
        self._view_pixmap: QPixmap | None = None # 原图缩放到当前 scale 的缓存，避免每帧重缩放
        self._view_pixmap_scale: float = -1.0
        self._stamp_cache: dict = {}             # 笔刷印章 LUT（按 radius_img, hardness 缓存）

        self._layer: np.ndarray | None = None        # H x W x 4 uint8
        self._original: np.ndarray | None = None     # H x W x 4 uint8，原始图（用于恢复模式参考色）
        self._tool: str = "erase"                    # "erase" | "paint" | "restore"
        self._brush_radius: int = 20                 # 屏幕像素（不随缩放变化）
        self._brush_hardness: float = 0.8            # 0..1
        self._brush_color: QColor = QColor(255, 0, 0, 255)
        self._pixel_mode: bool = False               # True=像素格模式（1x1对齐，不放大）
        self._drawing: bool = False
        self._cursor_pos: Optional[QPoint] = None
        # 笔刷缓动：上一次落笔的图像坐标
        self._last_image_pt: Optional[Tuple[int, int]] = None
        self._img_w = 0
        self._img_h = 0

        # 笔触节流：把待绘线段加入 pending，timer 触发统一绘制
        self._pending_segments: list[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self._pending_dirty = QRect()  # 待绘区域并集（图像坐标）
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(16)  # ~60 fps
        self._flush_timer.timeout.connect(self._flush_pending)

        # 在 _label 上叠画笔刷光标：覆盖 _label.paintEvent（在父类绘制后追加光标）
        self._label.paintEvent = self._label_paint_event  # type: ignore[assignment]

        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    def load(self, rgba: np.ndarray) -> None:
        """载入一张 RGBA 图作为当前编辑图层。"""
        if rgba.ndim == 2:
            rgba = np.stack([rgba] * 3, axis=-1)
        if rgba.shape[2] == 3:
            rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, dtype=np.uint8)])
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        self._original = rgba.copy()
        self._layer = rgba.copy()
        self._img_h, self._img_w = rgba.shape[:2]

        # 创建 canvas (ARGB32_Premultiplied)
        self._canvas = self._numpy_to_qimage(rgba)
        # 创建棋盘格背景缓存
        self._bg_pixmap = self._build_checker_pixmap(self._img_w, self._img_h)
        # 失效视图缓存
        self._view_pixmap = None
        self._view_pixmap_scale = -1.0

        # 临时设置一个与 canvas 等大的占位 pixmap，让 fit_to_view 能算出正确 scale
        placeholder = QPixmap(self._img_w, self._img_h)
        placeholder.fill(Qt.transparent)
        self._pixmap = placeholder
        self.info_label.setText(f"{self._img_w} × {self._img_h} px")
        self._scale = 1.0  # 若 fit_to_view 失败，回落到 1:1
        self.fit_to_view()
        self.stamp_status()

    def get_layer(self) -> Optional[np.ndarray]:
        return None if self._layer is None else self._layer.copy()

    def clear(self) -> None:
        super().clear()
        self._layer = None
        self._original = None
        self._canvas = None
        self._bg_pixmap = None
        self._view_pixmap = None
        self._drawing = False
        self._cursor_pos = None
        self._last_image_pt = None
        self._pending_segments.clear()
        self._pending_dirty = QRect()
        self._flush_timer.stop()

    def set_tool(self, tool: str) -> None:
        if tool not in ("erase", "paint", "restore"):
            return
        self._tool = tool
        self._label.update()

    def set_brush_radius(self, r: int) -> None:
        self._brush_radius = max(1, int(r))
        self._label.update()

    def set_brush_hardness(self, h: float) -> None:
        self._brush_hardness = float(max(0.0, min(1.0, h)))

    def set_brush_color(self, c: QColor) -> None:
        self._brush_color = QColor(c)

    def set_pixel_mode(self, on: bool) -> None:
        self._pixel_mode = on
        self._label.update()

    # ------------------------------------------------------------------
    # 资源转换 / 缓存
    # ------------------------------------------------------------------
    @staticmethod
    def _numpy_to_qimage(rgba: np.ndarray) -> QImage:
        h, w = rgba.shape[:2]
        # Format_ARGB32 在小端机器上字节序是 BGRA，QImage 会正确解释
        # Format_RGBA8888: QImage 字节顺序 = R-G-B-A，与 numpy RGBA 完全一致
        # （不要用 Format_ARGB32，在小端机器上 QImage 解释为 BGRA，会导致蓝/红互换）
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
        return qimg.copy()

    def _build_checker_pixmap(self, w: int, h: int) -> QPixmap:
        """一次性生成棋盘格背景，避免每帧重算。"""
        cell = 8
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                c = QColor(200, 200, 200) if ((y // cell) + (x // cell)) % 2 == 0 else QColor(120, 120, 120)
                p.fillRect(x, y, cell, cell, c)
        p.end()
        return pm

    def _get_scaled_canvas_pixmap(self) -> Optional[QPixmap]:
        """缩放缓存：同 scale 复用，避免每帧做一次高质量缩放。"""
        if self._canvas is None:
            return None
        if self._view_pixmap is None or abs(self._view_pixmap_scale - self._scale) > 1e-3:
            # canvas 显示的是像素级编辑结果，用 nearest 保持格子纯净
            # （小图放大时平滑插值会让格子边缘出现中间色）
            self._view_pixmap = QPixmap.fromImage(self._canvas).scaled(
                int(self._img_w * self._scale),
                int(self._img_h * self._scale),
                Qt.IgnoreAspectRatio,
                Qt.FastTransformation,
            )
            self._view_pixmap_scale = self._scale
        return self._view_pixmap

    def _get_scaled_bg_pixmap(self) -> Optional[QPixmap]:
        if self._bg_pixmap is None:
            return None
        # 棋盘格本身就是矢量意义不大，但用 FastTransformation 速度够快，且只在 scale 改变时算一次
        if not hasattr(self, "_bg_pixmap_scaled") or self._bg_pixmap_scaled is None or self._bg_pixmap_scale != self._scale:
            self._bg_pixmap_scaled = self._bg_pixmap.scaled(
                int(self._img_w * self._scale),
                int(self._img_h * self._scale),
                Qt.IgnoreAspectRatio,
                Qt.FastTransformation,
            )
            self._bg_pixmap_scale = self._scale
        return self._bg_pixmap_scaled

    # ------------------------------------------------------------------
    def _widget_to_image(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        if self._canvas is None or self._img_w == 0 or self._img_h == 0:
            return None
        local = self._label.mapFrom(self, pos)
        lw, lh = self._label.width(), self._label.height()
        sw = int(self._img_w * self._scale)
        sh = int(self._img_h * self._scale)
        lx = lw / 2 - sw / 2
        ly = lh / 2 - sh / 2
        x = local.x() - lx
        y = local.y() - ly
        if x < 0 or y < 0 or x >= sw or y >= sh:
            return None
        ix = max(0, min(self._img_w - 1, int(x / self._scale)))
        iy = max(0, min(self._img_h - 1, int(y / self._scale)))
        return ix, iy

    def _image_to_widget(self, ix: int, iy: int) -> Tuple[float, float]:
        lw, lh = self._label.width(), self._label.height()
        sw = int(self._img_w * self._scale)
        sh = int(self._img_h * self._scale)
        lx = lw / 2 - sw / 2
        ly = lh / 2 - sh / 2
        return lx + ix * self._scale, ly + iy * self._scale

    # ------------------------------------------------------------------
    # 笔触（高性能 QPainter 实现）
    # ------------------------------------------------------------------
    def _brush_radius_img(self) -> int:
        return max(1, int(round(self._brush_radius / self._scale)))

    def _get_brush_stamp(self) -> QImage:
        """返回一张笔刷"印章"图像：圆内带软边 alpha 渐变。

        缓存 key = (radius_img, hardness, tool, color_rgba)，避免每次绘制重算。
        """
        from PySide6.QtGui import QRadialGradient
        key = (
            self._brush_radius_img(),
            round(self._brush_hardness * 100) / 100,
            self._tool,
            self._brush_color.red(),
            self._brush_color.green(),
            self._brush_color.blue(),
            self._brush_color.alpha(),
        )
        stamp = self._stamp_cache.get(key)
        if stamp is not None:
            return stamp
        r = max(1, key[0])
        soft = max(0.0, 1.0 - key[1])
        inner_r = max(0, int(r * (1.0 - soft)))
        size = r * 2 + 2
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        if self._tool == "erase":
            # DestinationOut 模式：alpha 越高擦除越多；颜色无所谓，但 alpha=255 表示最强擦除
            rg = QRadialGradient(size / 2, size / 2, max(r, 1))
            rg.setColorAt(0.0, QColor(0, 0, 0, 255))
            stop_in = max(0.001, inner_r / max(r, 1))
            rg.setColorAt(stop_in, QColor(0, 0, 0, 255))
            rg.setColorAt(1.0, QColor(0, 0, 0, 0))
        elif self._tool == "paint":
            c = QColor(self._brush_color)
            rg = QRadialGradient(size / 2, size / 2, max(r, 1))
            rg.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(), 255))
            stop_in = max(0.001, inner_r / max(r, 1))
            rg.setColorAt(stop_in, QColor(c.red(), c.green(), c.blue(), 255))
            rg.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
        else:
            # restore 走专用路径，不应调用本方法
            rg = QRadialGradient(size / 2, size / 2, max(r, 1))
            rg.setColorAt(0.0, QColor(255, 255, 255, 255))
            rg.setColorAt(1.0, QColor(255, 255, 255, 255))
        p.setBrush(QBrush(rg))
        p.drawEllipse(1, 1, size - 2, size - 2)
        p.end()
        self._stamp_cache[key] = img
        return img

    def _queue_stroke_segment(self, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        self._pending_segments.append((a, b))
        # 扩展脏区域（图像坐标）
        # 像素格模式：固定 1 像素（精确到格子）
        r_img = 1 if self._pixel_mode else self._brush_radius_img()
        x0 = min(a[0], b[0]) - r_img - 1
        y0 = min(a[1], b[1]) - r_img - 1
        x1 = max(a[0], b[0]) + r_img + 2
        y1 = max(a[1], b[1]) + r_img + 2
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(self._img_w, x1); y1 = min(self._img_h, y1)
        seg_rect = QRect(x0, y0, x1 - x0, y1 - y0)
        if self._pending_dirty.isNull():
            self._pending_dirty = seg_rect
        else:
            self._pending_dirty = self._pending_dirty.united(seg_rect)

    def _flush_pending(self) -> None:
        """统一把待绘段画到 canvas，更新 _layer，并仅刷新 dirty 矩形。"""
        if not self._pending_segments or self._canvas is None:
            return

        # 像素格模式：numpy 直接操作，绕过 QPainter 软边
        if self._pixel_mode and self._layer is not None:
            self._flush_pending_pixel()
            return

        painter = QPainter(self._canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self._tool == "restore":
            self._restore_segments(painter)
        else:
            # 根据工具设置合成模式
            if self._tool == "erase":
                painter.setCompositionMode(QPainter.CompositionMode_DestinationOut)
            else:  # paint
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            # 用软笔刷印章（RGBA 渐变 alpha）dab 在每个笔触端点 + 沿线段均匀补点
            stamp = self._get_brush_stamp()
            r_img = self._brush_radius_img()
            step = max(1, r_img // 2)
            placed = []
            for a, b in self._pending_segments:
                dx = b[0] - a[0]; dy = b[1] - a[1]
                dist = int(np.hypot(dx, dy))
                n = max(1, dist // step)
                for i in range(n + 1):
                    t = i / n
                    ix = int(round(a[0] + dx * t))
                    iy = int(round(a[1] + dy * t))
                    placed.append((ix, iy))
            for ix, iy in placed:
                painter.drawImage(ix - r_img - 1, iy - r_img - 1, stamp)

        painter.end()

        # 把画布转 numpy（用于撤销栈、_layer 字段、导出）
        self._sync_layer_from_canvas()

        # 让画缓存失效
        self._view_pixmap = None
        self._view_pixmap_scale = -1.0

        # 仅刷新脏区域对应的 widget 矩形
        if not self._pending_dirty.isNull():
            self._repaint_label_dirty()

    def _repaint_label_dirty(self) -> None:
        if self._pending_dirty.isNull():
            return
        lw, lh = self._label.width(), self._label.height()
        sw = int(self._img_w * self._scale)
        sh = int(self._img_h * self._scale)
        lx = lw / 2 - sw / 2
        ly = lh / 2 - sh / 2
        d = self._pending_dirty
        x0 = lx + d.left() * self._scale
        y0 = ly + d.top() * self._scale
        x1 = lx + (d.right() + 1) * self._scale
        y1 = ly + (d.bottom() + 1) * self._scale
        self._label.update(QRectF(x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4).toRect())
        self._pending_segments.clear()
        self._pending_dirty = QRect()

    def _flush_pending_pixel(self) -> None:
        """像素格模式：numpy 直接操作每个 1x1 像素，绕过 QPainter 软边插值。"""
        if not self._pending_segments or self._layer is None:
            return
        r, g, b, a = (
            self._brush_color.red(),
            self._brush_color.green(),
            self._brush_color.blue(),
            self._brush_color.alpha(),
        )
        covered: set = set()

        for a_pt, b_pt in self._pending_segments:
            for px in self._line_pixels(a_pt[0], a_pt[1], b_pt[0], b_pt[1]):
                covered.add(px)

        if not covered:
            self._pending_segments.clear()
            self._pending_dirty = QRect()
            return

        layer = self._layer
        if self._tool == "erase":
            for (cx, cy) in covered:
                if 0 <= cx < self._img_w and 0 <= cy < self._img_h:
                    layer[cy, cx, 3] = 0
        elif self._tool == "paint":
            for (cx, cy) in covered:
                if 0 <= cx < self._img_w and 0 <= cy < self._img_h:
                    layer[cy, cx] = [r, g, b, a]
        elif self._tool == "restore" and self._original is not None:
            for (cx, cy) in covered:
                if 0 <= cx < self._img_w and 0 <= cy < self._img_h:
                    layer[cy, cx] = self._original[cy, cx]

        # 重建整个 canvas（避免 partial-update 的 QPainter/QImage 边界问题）
        self._canvas = self._numpy_to_qimage(layer)
        self._view_pixmap = None
        self._view_pixmap_scale = -1.0
        self._repaint_label_dirty()

    def _line_pixels(self, x0: int, y0: int, x1: int, y1: int):
        """Bresenham 追踪线段覆盖的所有整数坐标（generator）。"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        cx, cy = x0, y0
        while True:
            yield (cx, cy)
            if cx == x1 and cy == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

    def _sync_dirty_rect_to_canvas(self) -> None:
        """只把 _layer 的 dirty 区域写回 _canvas（QImage）。"""
        if self._canvas is None or self._layer is None:
            return
        d = self._pending_dirty
        if d.isNull():
            return
        x0 = max(0, d.left())
        y0 = max(0, d.top())
        x1 = min(self._img_w, d.right() + 1)
        y1 = min(self._img_h, d.bottom() + 1)
        if x0 >= x1 or y0 >= y1:
            return
        roi = self._layer[y0:y1, x0:x1]
        roi_qimg = self._numpy_to_qimage(roi)
        painter = QPainter(self._canvas)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawImage(QPoint(x0, y0), roi_qimg)
        painter.end()

    def _restore_segments(self, painter: QPainter) -> None:
        """恢复模式：把笔刷范围内的像素从 self._original 拷贝回来。

        实现：
        1) DestinationOut 用圆头粗笔刷清掉笔触覆盖区 alpha（圆头自然软边）
        2) SourceOver 贴回 original 的 ROI（中心 alpha=255，边缘靠步骤 1 的 antialias）
        """
        if self._original is None:
            return
        d = self._pending_dirty
        if d.isNull():
            return
        x0, y0, x1, y1 = d.left(), d.top(), d.right() + 1, d.bottom() + 1
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(self._img_w, x1); y1 = min(self._img_h, y1)
        if x0 >= x1 or y0 >= y1:
            return

        # 步骤 1: DestinationOut 清 alpha（圆头笔触 + antialias 给出软边）
        painter.setCompositionMode(QPainter.CompositionMode_DestinationOut)
        pen_clear = QPen(QColor(0, 0, 0, 255), self._brush_radius_img() * 2,
                         Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        pen_clear.setCosmetic(False)
        painter.setPen(pen_clear)
        for a, b in self._pending_segments:
            painter.drawLine(a[0], a[1], b[0], b[1])

        # 步骤 2: SourceOver 贴回 original（中心 alpha=255）
        roi = self._original[y0:y1, x0:x1]
        roi = np.ascontiguousarray(roi)
        roi_qimg = self._numpy_to_qimage(roi)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawImage(QPoint(x0, y0), roi_qimg)

    def _sync_layer_from_canvas(self) -> None:
        """把 QImage 同步到 numpy（H x W x 4 uint8）。"""
        if self._canvas is None:
            return
        w, h = self._canvas.width(), self._canvas.height()
        # QImage.bits() 返回 bytes-like，构造 numpy 视图（自动拷贝以脱离 QImage 生命周期）
        ptr = self._canvas.bits()
        # PySide6 下 bits() 是 sip.voidptr，要先 tobytes() 或 cast
        try:
            buf = bytes(ptr)  # type: ignore[arg-type]
        except TypeError:
            # 老接口：ptr 已经是 memoryview
            buf = ptr.tobytes() if hasattr(ptr, "tobytes") else bytes(memoryview(ptr))
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4).copy()
        # RGBA8888: QImage 字节顺序 = R-G-B-A，与 numpy RGBA 完全一致，无需交换
        # 所以直接用 reshape 得到的数组就是正确的 RGBA
        bgra = arr  # 复用变量名但实际是 RGBA
        rgba = bgra.copy()
        self._layer = rgba

    # ------------------------------------------------------------------
    # 鼠标事件
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._canvas is not None:
            pt = self._widget_to_image(event.position().toPoint())
            if pt is not None:
                self._drawing = True
                self._last_image_pt = pt
                # 立即画一个点（按下时）
                self._queue_stroke_segment(pt, pt)
                self._flush_pending()
                self.stroke_started.emit()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self._cursor_pos = event.position().toPoint()
        pt = self._widget_to_image(self._cursor_pos)
        if pt is not None:
            self.cursor_moved.emit(pt[0], pt[1])
        else:
            self.cursor_moved.emit(-1, -1)
        if self._drawing and self._last_image_pt is not None and self._canvas is not None and pt is not None:
            self._queue_stroke_segment(self._last_image_pt, pt)
            self._last_image_pt = pt
            # 节流：启动定时器，到点再统一画
            if not self._flush_timer.isActive():
                self._flush_timer.start()
        self._label.update()  # 更新光标位置

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drawing:
            # 立刻 flush 待绘段（不等定时器）
            self._flush_timer.stop()
            self._flush_pending()
            self._drawing = False
            self._last_image_pt = None
            self.stroke_finished.emit()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._cursor_pos = None
        self.cursor_moved.emit(-1, -1)
        self._label.update()
        super().leaveEvent(event)

    def _apply(self) -> None:
        """重写 ImageView._apply：清空 _label.pixmap，由我们的 paintEvent 接管绘制。"""
        if self._canvas is None:
            return
        self._label.setPixmap(QPixmap())
        self._label.resize(int(self._img_w * self._scale), int(self._img_h * self._scale))
        self._label.update()

    def wheelEvent(self, event) -> None:
        super().wheelEvent(event)
        # scale 变了：失效缓存
        self._view_pixmap = None
        self._view_pixmap_scale = -1.0
        self._bg_pixmap_scaled = None
        self._bg_pixmap_scale = -1.0
        # 清掉 ImageView 设到 _label 上的旧 pixmap，避免画两次
        self._label.setPixmap(QPixmap())
        self._label.update()

    # ------------------------------------------------------------------
    # 撤销 / 重做 同步（暴露给主控件）
    # ------------------------------------------------------------------
    def replace_layer(self, rgba: np.ndarray) -> None:
        """用一张 RGBA 图替换当前画布和 _layer（用于撤销 / 重做）。"""
        if rgba.ndim == 2:
            rgba = np.stack([rgba] * 3, axis=-1)
        if rgba.shape[2] == 3:
            rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, dtype=np.uint8)])
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        if rgba.shape[0] != self._img_h or rgba.shape[1] != self._img_w:
            return
        self._layer = rgba.copy()
        self._canvas = self._numpy_to_qimage(rgba)
        self._view_pixmap = None
        self._view_pixmap_scale = -1.0
        self._label.update()

    # ------------------------------------------------------------------
    def stamp_status(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 在 _label 上画：背景 + canvas + 笔刷光标
    # ------------------------------------------------------------------
    def _label_paint_event(self, event) -> None:
        if self._canvas is None:
            # 默认走父类（显示占位）
            from PySide6.QtWidgets import QLabel
            QLabel.paintEvent(self._label, event)
            return

        # 1. 画背景（棋盘格）
        bg = self._get_scaled_bg_pixmap()
        if bg is not None:
            lw, lh = self._label.width(), self._label.height()
            sw = bg.width(); sh = bg.height()
            lx = int(lw / 2 - sw / 2)
            ly = int(lh / 2 - sh / 2)
            painter = QPainter(self._label)
            painter.setClipRect(event.rect())
            painter.drawPixmap(lx, ly, bg)
            # 2. 画 canvas（叠加到背景上）
            cv = self._get_scaled_canvas_pixmap()
            if cv is not None:
                painter.drawPixmap(lx, ly, cv)
            # 3. 画笔刷光标
            if self._cursor_pos is not None:
                img_pt = self._widget_to_image(self._cursor_pos)
                if img_pt is not None:
                    sx, sy = self._image_to_widget(img_pt[0], img_pt[1])
                    ring_color = {
                        "erase":   QColor(255, 80, 80),
                        "paint":   QColor(80, 200, 255),
                        "restore": QColor(255, 200, 80),
                    }.get(self._tool, QColor(255, 255, 255))
                    pen = QPen(ring_color, 2)
                    pen.setCosmetic(True)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(QPointF(sx, sy), self._brush_radius, self._brush_radius)
            painter.end()
        else:
            from PySide6.QtWidgets import QLabel
            QLabel.paintEvent(self._label, event)


# ---------------------------------------------------------------------------
# 主控件
# ---------------------------------------------------------------------------

class ManualEditorWidget(QWidget):
    """手动编辑模块：打开 / 从暂存区载入图，画笔 / 橡皮擦涂抹，导出或加入暂存区。"""

    _buf: Optional[ImageBuffer] = None

    @staticmethod
    def set_buffer_ref(buf: Optional[ImageBuffer]) -> None:
        ManualEditorWidget._buf = buf

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buf: Optional[ImageBuffer] = ManualEditorWidget._buf
        self.last_saved_path: Optional[str] = None

        # ---- 撤销/重做栈 ----
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._max_undo = 30

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- 顶部工具栏 ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.btn_open = QPushButton("打开图片…")
        self.btn_open.clicked.connect(self._on_open)
        toolbar.addWidget(self.btn_open)

        self.btn_load_buffer = QPushButton("从暂存区载入")
        self.btn_load_buffer.clicked.connect(self._on_load_from_buffer)
        toolbar.addWidget(self.btn_load_buffer)

        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._on_clear)
        toolbar.addWidget(self.btn_clear)

        toolbar.addSpacing(20)

        # 工具：橡皮擦 / 画笔 / 恢复（按钮互斥）
        self.tool_buttons = QButtonGroup(self)
        self.tool_buttons.setExclusive(True)
        self.btn_erase = QPushButton("🧽 橡皮擦")
        self.btn_erase.setCheckable(True)
        self.btn_erase.setChecked(True)
        self.tool_buttons.addButton(self.btn_erase, 0)
        toolbar.addWidget(self.btn_erase)

        self.btn_paint = QPushButton("🖌️ 画笔")
        self.btn_paint.setCheckable(True)
        self.tool_buttons.addButton(self.btn_paint, 1)
        toolbar.addWidget(self.btn_paint)

        self.btn_restore = QPushButton("🔁 恢复（回退到原图）")
        self.btn_restore.setCheckable(True)
        self.tool_buttons.addButton(self.btn_restore, 2)
        toolbar.addWidget(self.btn_restore)

        self.tool_buttons.idClicked.connect(self._on_tool_changed)

        toolbar.addSpacing(20)

        self.chk_pixel_mode = QCheckBox("像素格模式")
        self.chk_pixel_mode.setToolTip(
            "开启后笔刷以 1×1 像素为最小单位，坐标对齐网格，适合像素画编辑。\n"
            "关闭时为任意模式，笔刷有软边，可画自由曲线。"
        )
        self.chk_pixel_mode.stateChanged.connect(self._on_pixel_mode_changed)
        toolbar.addWidget(self.chk_pixel_mode)

        toolbar.addSpacing(20)

        self.lbl_color = QLabel("画笔颜色:")
        toolbar.addWidget(self.lbl_color)
        self.color_btn = ColorSwatchButton(QColor(255, 0, 0))
        self.color_btn.setColor(self.color_btn.color())
        self.color_btn.clicked  # 信号已绑定
        toolbar.addWidget(self.color_btn)
        # 监听颜色变化
        self._current_color = QColor(255, 0, 0)
        self.color_btn.clicked.connect(self._on_color_picked)

        self.btn_picker = QPushButton("🎯 吸管（从图上取色）")
        self.btn_picker.setCheckable(True)
        self.btn_picker.toggled.connect(self._on_picker_toggled)
        toolbar.addWidget(self.btn_picker)

        toolbar.addStretch(1)

        self.btn_undo = QPushButton("↶ 撤销")
        self.btn_undo.clicked.connect(self._on_undo)
        toolbar.addWidget(self.btn_undo)
        self.btn_redo = QPushButton("↷ 重做")
        self.btn_redo.clicked.connect(self._on_redo)
        toolbar.addWidget(self.btn_redo)

        root.addLayout(toolbar)

        # ---- 第二行：笔刷参数 ----
        param_row = QHBoxLayout()
        param_row.setSpacing(8)

        param_row.addWidget(QLabel("笔刷:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 200)
        self.brush_slider.setValue(20)
        self.brush_slider.setFixedWidth(220)
        self.brush_slider.valueChanged.connect(self._on_brush_changed)
        param_row.addWidget(self.brush_slider)
        self.brush_value = QLabel("20 px")
        param_row.addWidget(self.brush_value)

        param_row.addSpacing(20)

        param_row.addWidget(QLabel("硬度:"))
        self.hardness_slider = QSlider(Qt.Horizontal)
        self.hardness_slider.setRange(0, 100)
        self.hardness_slider.setValue(80)
        self.hardness_slider.setFixedWidth(160)
        self.hardness_slider.valueChanged.connect(self._on_hardness_changed)
        param_row.addWidget(self.hardness_slider)
        self.hardness_value = QLabel("0.80")
        param_row.addWidget(self.hardness_value)

        param_row.addStretch(1)

        self.btn_reset_brush = QPushButton("重置笔刷到中心")
        self.btn_reset_brush.clicked.connect(self._on_reset_brush)
        param_row.addWidget(self.btn_reset_brush)

        root.addLayout(param_row)

        # ---- 预览面板 ----
        self.view = EditableImageView("画布（拖动鼠标涂抹，滚轮缩放）")
        self.view.cursor_moved.connect(self._on_cursor_moved)
        self.view.stroke_started.connect(self._on_stroke_started)
        self.view.stroke_finished.connect(self._on_stroke_finished)
        root.addWidget(self.view, 1)

        # ---- 底部动作栏 ----
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_export_png = QPushButton("导出 PNG")
        self.btn_export_png.clicked.connect(self._export_png)
        action_row.addWidget(self.btn_export_png)

        self.btn_export_rgb = QPushButton("导出 RGB（合成到白底）")
        self.btn_export_rgb.clicked.connect(self._export_rgb)
        action_row.addWidget(self.btn_export_rgb)

        self.btn_add_to_tray = QPushButton("加入暂存区")
        self.btn_add_to_tray.clicked.connect(self._push_to_buffer)
        action_row.addWidget(self.btn_add_to_tray)

        action_row.addStretch(1)

        self.lbl_info = QLabel("提示：先用「打开图片」或「从暂存区载入」载入一张图")
        self.lbl_info.setStyleSheet("color: #888;")
        action_row.addWidget(self.lbl_info)

        root.addLayout(action_row)

        # ---- 快捷键 ----
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._on_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._on_redo)
        QShortcut(QKeySequence("["), self, activated=lambda: self.brush_slider.setValue(max(1, self.brush_slider.value() - 2)))
        QShortcut(QKeySequence("]"), self, activated=lambda: self.brush_slider.setValue(min(200, self.brush_slider.value() + 2)))

        self._update_button_states()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def _update_button_states(self) -> None:
        has_image = self.view._layer is not None
        for w in (
            self.btn_clear,
            self.btn_export_png,
            self.btn_export_rgb,
            self.btn_add_to_tray,
            self.btn_paint,
            self.btn_erase,
            self.btn_restore,
            self.brush_slider,
            self.hardness_slider,
            self.btn_reset_brush,
            self.color_btn,
            self.btn_picker,
            self.chk_pixel_mode,
        ):
            w.setEnabled(has_image)
        self.btn_load_buffer.setEnabled(self._buf is not None)
        self.btn_undo.setEnabled(has_image and len(self._undo_stack) > 0)
        self.btn_redo.setEnabled(has_image and len(self._redo_stack) > 0)

    # ------------------------------------------------------------------
    # 工具 / 颜色 / 笔刷
    # ------------------------------------------------------------------
    def _on_tool_changed(self, _id: int) -> None:
        if self.btn_erase.isChecked():
            self.view.set_tool("erase")
        elif self.btn_paint.isChecked():
            self.view.set_tool("paint")
        elif self.btn_restore.isChecked():
            self.view.set_tool("restore")
        # 切到画笔时把吸管自动关闭
        if not self.btn_picker.isChecked():
            return
        if self.btn_paint.isChecked() or self.btn_erase.isChecked() or self.btn_restore.isChecked():
            self.btn_picker.setChecked(False)

    def _on_pixel_mode_changed(self, state: int) -> None:
        on = state == Qt.CheckState.Checked.value
        self.view.set_pixel_mode(on)
        # 像素格模式下：硬度强制 1.0（纯硬边），笔刷半径固定为 1
        if on:
            self.brush_slider.setEnabled(False)
            self.hardness_slider.setEnabled(False)
            self.brush_slider.setToolTip("像素格模式下笔刷大小固定为 1 像素")
            self.hardness_slider.setToolTip("像素格模式下硬度固定为 1.0（硬边）")
        else:
            self.brush_slider.setEnabled(True)
            self.hardness_slider.setEnabled(True)
            self.brush_slider.setToolTip("")
            self.hardness_slider.setToolTip("")

    def _on_color_picked(self) -> None:
        c = self.color_btn.color()
        self._current_color = c
        self.view.set_brush_color(c)
        # 自动切到画笔工具
        if not self.btn_paint.isChecked():
            self.btn_paint.setChecked(True)
            self.btn_erase.setChecked(False)
            self.btn_restore.setChecked(False)
            self._on_tool_changed(1)

    def _on_picker_toggled(self, on: bool) -> None:
        # 吸管模式只切坐标，并不真的去吸颜色（鼠标按下事件时取色）
        self._picker_active = on
        if on:
            self.btn_paint.setChecked(False)
            self.btn_erase.setChecked(False)
            self.btn_restore.setChecked(False)

    def _on_brush_changed(self, v: int) -> None:
        self.view.set_brush_radius(v)
        self.brush_value.setText(f"{v} px")

    def _on_hardness_changed(self, v: int) -> None:
        h = v / 100.0
        self.view.set_brush_hardness(h)
        self.hardness_value.setText(f"{h:.2f}")

    def _on_reset_brush(self) -> None:
        self.brush_slider.setValue(20)
        self.hardness_slider.setValue(80)

    # ------------------------------------------------------------------
    # 文件
    # ------------------------------------------------------------------
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            pil = Image.open(path)
            pil = pil.convert("RGBA")
            arr = np.array(pil, dtype=np.uint8)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "打开失败", str(exc))
            return
        self.view.load(arr)
        self.last_saved_path = path
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._push_undo()
        self.lbl_info.setText(f"已载入: {Path(path).name}")
        self._update_button_states()

    def _on_load_from_buffer(self) -> None:
        if self._buf is None:
            QMessageBox.warning(self, "暂存区不可用", "暂存区未初始化")
            return
        latest = self._buf.items()[-1] if self._buf.items() else None
        if latest is None:
            QMessageBox.information(self, "暂存区为空", "暂存区里没有图片")
            return
        self.load_from_buffer(latest["image"], source_tab=latest.get("source_tab", ""))

    def load_from_buffer(self, image: np.ndarray, source_tab: str = "") -> None:
        """主程序在右侧缩略图被双击时回调此方法，把图片载入当前画布。"""
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 3:
            arr = np.dstack([arr, np.full(arr.shape[:2], 255, dtype=np.uint8)])
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        self.view.load(arr)
        self.lbl_info.setText(f"已从暂存区载入（{source_tab}）")
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._push_undo()
        self._update_button_states()

    def _on_clear(self) -> None:
        self.view.clear()
        self.last_saved_path = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.lbl_info.setText("提示：先用「打开图片」或「从暂存区载入」载入一张图")
        self._update_button_states()

    # ------------------------------------------------------------------
    # 鼠标交互挂钩
    # ------------------------------------------------------------------
    def _on_cursor_moved(self, x: int, y: int) -> None:
        if x < 0 or y < 0:
            self.lbl_info.setText(f"已载入图像，{self.view._img_w}×{self.view._img_h}")
            return
        # 在状态栏/标签里展示当前坐标和颜色
        layer = self.view._layer
        if layer is not None:
            r, g, b, a = layer[y, x].tolist()
            self.lbl_info.setText(
                f"坐标 ({x},{y})  RGBA=({r},{g},{b},{a})  |  工具="
                + ("橡皮擦" if self.btn_erase.isChecked() else "画笔" if self.btn_paint.isChecked() else "恢复")
            )

    def _on_stroke_started(self) -> None:
        # 一笔开始前推一份快照进 undo 栈
        self._push_undo()

    def _on_stroke_finished(self) -> None:
        # 笔结束时更新按钮状态（undo 已有内容）
        self._update_button_states()

    # ------------------------------------------------------------------
    # 吸管：从图上取色
    # ------------------------------------------------------------------
    def _maybe_pick_color(self, view_pos: QPoint) -> None:
        if not getattr(self, "_picker_active", False):
            return
        pt = self.view._widget_to_image(view_pos)
        if pt is None:
            return
        x, y = pt
        # 取 original 的颜色（避免取到笔刷影响过的像素）
        if self.view._original is not None:
            r, g, b, a = self.view._original[y, x].tolist()
        elif self.view._layer is not None:
            r, g, b, a = self.view._layer[y, x].tolist()
        else:
            return
        c = QColor(r, g, b, a)
        self.color_btn.setColor(c)
        self._current_color = c
        self.view.set_brush_color(c)
        self.btn_picker.setChecked(False)
        if not self.btn_paint.isChecked():
            self.btn_paint.setChecked(True)
            self.btn_erase.setChecked(False)
            self.btn_restore.setChecked(False)
            self._on_tool_changed(1)

    def mousePressEvent(self, event) -> None:
        if getattr(self, "_picker_active", False) and event.button() == Qt.LeftButton:
            self._maybe_pick_color(event.position().toPoint())
            return
        # 把事件发给 view
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if getattr(self, "_picker_active", False):
            self.view.setCursor(Qt.CrossCursor)
        super().mouseMoveEvent(event)

    # ------------------------------------------------------------------
    # 撤销 / 重做
    # ------------------------------------------------------------------
    def _push_undo(self) -> None:
        if self.view._layer is None:
            return
        snap = self.view._layer.copy()
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_button_states()

    def _on_undo(self) -> None:
        if not self._undo_stack:
            return
        cur = self.view._layer
        if cur is not None:
            self._redo_stack.append(cur.copy())
        snap = self._undo_stack.pop()
        self.view.replace_layer(snap)
        self._update_button_states()

    def _on_redo(self) -> None:
        if not self._redo_stack:
            return
        cur = self.view._layer
        if cur is not None:
            self._undo_stack.append(cur.copy())
        snap = self._redo_stack.pop()
        self.view.replace_layer(snap)
        self._update_button_states()

    # ------------------------------------------------------------------
    # 导出 / 加入暂存区
    # ------------------------------------------------------------------
    def _current(self) -> Optional[np.ndarray]:
        return self.view.get_layer()

    def _export_png(self) -> None:
        cur = self._current()
        if cur is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 PNG", self.last_saved_path or "", "PNG (*.png)"
        )
        if not path:
            return
        try:
            Image.fromarray(cur, mode="RGBA").save(path, "PNG")
            self.last_saved_path = path
            self.lbl_info.setText(f"已导出 {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))

    def _export_rgb(self) -> None:
        cur = self._current()
        if cur is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出图片", self.last_saved_path or "", "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        try:
            rgb = self._flatten_on_white(cur)
            pil = Image.fromarray(rgb)
            if path.lower().endswith(".jpg"):
                pil.save(path, "JPEG", quality=95)
            else:
                pil.save(path, "PNG")
            self.last_saved_path = path
            self.lbl_info.setText(f"已导出 {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))

    @staticmethod
    def _flatten_on_white(rgba: np.ndarray) -> np.ndarray:
        a = rgba[:, :, 3:4].astype(np.float32) / 255.0
        fg = rgba[:, :, :3].astype(np.float32)
        out = fg * a + 255.0 * (1.0 - a)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _push_to_buffer(self) -> None:
        if self._buf is None:
            QMessageBox.warning(self, "暂存区不可用", "暂存区未初始化")
            return
        cur = self._current()
        if cur is None:
            QMessageBox.warning(self, "无可用结果", "请先打开或载入一张图")
            return
        self._buf.push(cur, source_tab="手动编辑")
        self.lbl_info.setText("已加入暂存区")
