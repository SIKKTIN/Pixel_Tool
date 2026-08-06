"""
图片裁剪模块：将任意分辨率的图片裁剪到指定分辨率。

用法：
    from image_crop import ImageCropWidget
    ImageCropWidget.set_buffer_ref(buf)
    tabs.addTab(ImageCropWidget(), "🎯 裁剪到目标尺寸")
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QPointF, QRect, QRectF, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGraphicsPixmapItem,
    QGraphicsItem,
    QGraphicsItemGroup,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from PIL import Image
from PIL.ImageQt import ImageQt

from desktop_app import ImageBuffer


# ---------------------------------------------------------------------------
# 棋盘格背景（用于 CropView 透明区域显示）
# ---------------------------------------------------------------------------

_CHECKER_PIXMAP: QPixmap | None = None


def _make_checker_brush() -> QBrush:
    """生成一个 16×16 棋盘格 pixmap 作为 viewport 背景。

    颜色采用与项目其他模块一致的深灰色调（#2a2a2a / #222），与图像内容
    形成清晰对比，方便看清图像实际范围。
    """
    global _CHECKER_PIXMAP
    if _CHECKER_PIXMAP is not None and not _CHECKER_PIXMAP.isNull():
        return QBrush(_CHECKER_PIXMAP)

    size = 16
    pm = QPixmap(size, size)
    pm.fill(QColor("#2a2a2a"))
    p = QPainter(pm)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#222222"))
    p.drawRect(0, 0, size // 2, size // 2)
    p.drawRect(size // 2, size // 2, size // 2, size // 2)
    p.end()
    _CHECKER_PIXMAP = pm
    return QBrush(pm)


def _xp_border_pen() -> QPen:
    """图片边界的 1px 描边（半透明灰色，缩放时不会变粗）。"""
    pen = QPen(QColor(255, 255, 255, 70))
    pen.setCosmetic(True)
    pen.setWidth(1)
    return pen


# ---------------------------------------------------------------------------
# PIL -> QPixmap
# ---------------------------------------------------------------------------

def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    return QPixmap.fromImage(ImageQt(img))


def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 3:
        arr = np.dstack([arr, np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)])
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


# ---------------------------------------------------------------------------
# 占位符（纯QGraphicsItem，无widget）
# ---------------------------------------------------------------------------

class PlaceholderText(QGraphicsSimpleTextItem):
    """纯图形占位文本。"""

    def __init__(self, text: str, scene: QGraphicsScene, x: float = 0, y: float = 0) -> None:
        super().__init__(text)
        self.setFont(QFont("Segoe UI", 14))
        self.setBrush(QBrush(QColor("#555555")))
        self.setZValue(0)
        self.setPos(x, y)
        scene.addItem(self)


# ---------------------------------------------------------------------------
# 单个裁剪手柄
# ---------------------------------------------------------------------------

HANDLE_SIZE = 6


class HandleItem(QGraphicsRectItem):
    """可拖拽调整大小的角点/边中点手柄。

    通过 ItemIgnoresTransformations 标志让手柄始终保持屏幕像素尺寸，
    不管图片缩放比例如何都清晰可见且大小恒定。
    """

    def __init__(self, parent: QGraphicsItem | None = None) -> None:
        super().__init__(
            -HANDLE_SIZE / 2, -HANDLE_SIZE / 2,
            HANDLE_SIZE, HANDLE_SIZE,
            parent,
        )
        self.setBrush(QBrush(QColor(80, 220, 140)))
        pen = QPen(QColor(0, 60, 30), 1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(20)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QBrush(QColor(150, 255, 180)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QBrush(QColor(80, 220, 140)))
        super().hoverLeaveEvent(event)


# ---------------------------------------------------------------------------
# 裁剪框（边框 + 8个手柄）。遮罩由 view 在 paintEvent 中按屏幕坐标绘制。
# ---------------------------------------------------------------------------

class CropBox(QGraphicsItemGroup):
    """裁剪选区：金色边框 + 8个手柄。

    遮罩不在这里绘制，由 CropView.paintEvent 直接在 view 坐标系下绘制，
    这样无论图像大小、缩放比例如何都正确。
    """

    MIN_SIZE = 10

    def __init__(self, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.setHandlesChildEvents(False)
        self.setFlag(QGraphicsItem.ItemClipsChildrenToShape, False)
        self.setZValue(10)

        # 金色边框（双层：外黑 + 内黄，确保小选区也清晰）
        self._border_outer = QGraphicsRectItem(self)
        self._border_outer.setBrush(Qt.NoBrush)
        pen_outer = QPen(QColor(0, 0, 0, 220), 2)
        pen_outer.setCosmetic(True)
        self._border_outer.setPen(pen_outer)
        self._border_outer.setZValue(11)

        self._border = QGraphicsRectItem(self)
        self._border.setBrush(Qt.NoBrush)
        pen = QPen(QColor(80, 220, 140), 1)
        pen.setCosmetic(True)
        self._border.setPen(pen)
        self._border.setZValue(12)

        # 8个手柄
        self._handles: dict[str, HandleItem] = {}
        for name in ("nw", "n", "ne", "w", "e", "sw", "s", "se"):
            h = HandleItem(self)
            self._handles[name] = h

        self._rect = QRectF()
        self.hide()

    def setRect(self, r: QRectF) -> None:
        self._rect = r.normalized()
        bx = self._rect.x()
        by = self._rect.y()
        bw = self._rect.width()
        bh = self._rect.height()

        if bw < 1 or bh < 1:
            self.hide()
            return
        self.show()

        # 边框与手柄使用 CropBox 的局部坐标，先把 group 自身的 pos 移到矩形左上角
        self.setPos(bx, by)
        self._border_outer.setRect(0, 0, bw, bh)
        self._border.setRect(0, 0, bw, bh)

        # 手柄位置（cosmetic pen 始终保持屏幕像素尺寸）
        positions = {
            "nw": (0,     0),
            "n":  (bw/2,  0),
            "ne": (bw,    0),
            "w":  (0,     bh/2),
            "e":  (bw,    bh/2),
            "sw": (0,     bh),
            "s":  (bw/2,  bh),
            "se": (bw,    bh),
        }
        for name, (hx, hy) in positions.items():
            self._handles[name].setPos(hx, hy)

    def set_handles_visible(self, visible: bool) -> None:
        for h in self._handles.values():
            h.setVisible(visible)

    def rect(self) -> QRectF:
        return self._rect


# ---------------------------------------------------------------------------
# 原图视图：支持拖拽选区、Ctrl+滚轮缩放
# ---------------------------------------------------------------------------

class CropView(QGraphicsView):
    """显示原图 + 可视化裁剪选区。"""

    selection_changed = Signal(QRect)

    DRAG_NONE   = 0
    DRAG_MOVE   = 1
    DRAG_HANDLE = 2

    # 编辑模式
    MODE_RESIZE = "resize"  # 调整大小（默认）：手柄拖拽 + 空白拉框
    MODE_MOVE   = "move"    # 移动位置：只整体平移，禁用手柄和拉框

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        # 棋盘格背景，方便看出透明区域；与其他模块的暗色一致
        self.setBackgroundBrush(_make_checker_brush())
        self.setDragMode(QGraphicsView.NoDrag)
        # 任何场景变化都强制重画整个 viewport，保证 screen-layer 遮罩同步刷新
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

        self._pix_item: QGraphicsPixmapItem | None = None
        self._crop_box: CropBox | None = None
        self._image_border: QGraphicsRectItem | None = None
        self._image_border_inner: QGraphicsRectItem | None = None
        self._sel_rect = QRectF()
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 32.0
        self._image_size = (0, 0)
        self._mode = self.MODE_RESIZE

        # 拖拽状态
        self._drag_mode = self.DRAG_NONE
        self._drag_handle = ""
        self._drag_start = QPointF()
        self._drag_rect_start = QRectF()

        # 占位
        PlaceholderText("(无图片)", self._scene, 180, 180)
        self._scene.setSceneRect(0, 0, 500, 400)

    def load_image(self, arr: np.ndarray) -> None:
        """加载 RGBA numpy 数组。"""
        self._scene.clear()
        self._pix_item = None
        self._crop_box = None
        self._placeholder = None

        h, w = arr.shape[:2]
        self._image_size = (w, h)

        pix = pil_to_qpixmap(np_to_pil(arr))
        self._pix_item = QGraphicsPixmapItem(pix)
        self._pix_item.setZValue(0)
        self._scene.addItem(self._pix_item)

        # 图像边界框：醒目的黄色描边 + 外侧黑色描边（双色），即便缩很小也能看清
        self._image_border = QGraphicsRectItem(QRectF(0, 0, w, h))
        self._image_border.setZValue(1)
        # 用复合笔画：底层黑（外晕）+ 上层亮黄（主边）
        outer_pen = QPen(QColor(0, 0, 0, 200))
        outer_pen.setWidthF(5.0)
        outer_pen.setCosmetic(True)
        self._image_border.setPen(outer_pen)
        self._image_border.setBrush(Qt.NoBrush)
        self._scene.addItem(self._image_border)

        # 内层亮黄描边，盖在外黑边之上做主边
        inner_pen = QPen(QColor(255, 215, 0, 255))  # gold
        inner_pen.setWidthF(2.5)
        inner_pen.setCosmetic(True)
        self._image_border_inner = QGraphicsRectItem(QRectF(0, 0, w, h))
        self._image_border_inner.setZValue(1.1)
        self._image_border_inner.setPen(inner_pen)
        self._image_border_inner.setBrush(Qt.NoBrush)
        self._scene.addItem(self._image_border_inner)

        self._crop_box = CropBox()
        self._scene.addItem(self._crop_box)
        self._crop_box.set_handles_visible(self._mode == self.MODE_RESIZE)

        self._sel_rect = QRectF(0, 0, w, h)
        self._crop_box.setRect(self._sel_rect)

        # 给 scene 一个较大的画布，避免图片边缘出现奇怪的边框
        pad = 64
        self._scene.setSceneRect(-pad, -pad, w + 2 * pad, h + 2 * pad)
        self.fit_to_view()

    def clear(self) -> None:
        self._scene.clear()
        self._pix_item = None
        self._crop_box = None
        self._image_border = None
        self._image_border_inner = None
        self._placeholder = None
        PlaceholderText("(无图片)", self._scene, 180, 180)
        self._scene.setSceneRect(0, 0, 500, 400)

    def set_selection(self, rect: QRect) -> None:
        if self._crop_box is None:
            return
        r = QRectF(rect).normalized()
        r = r.intersected(QRectF(0, 0, *self._image_size))
        self._sel_rect = r
        self._crop_box.setRect(r)
        self.viewport().update()
        self.selection_changed.emit(r.toRect())

    def set_mode(self, mode: str) -> None:
        """切换编辑模式：MODE_RESIZE / MODE_MOVE."""
        if mode not in (self.MODE_RESIZE, self.MODE_MOVE):
            return
        self._mode = mode
        if self._crop_box is not None:
            self._crop_box.set_handles_visible(mode == self.MODE_RESIZE)
        self.viewport().update()

    def selection_rect(self) -> QRect:
        return self._sel_rect.toRect()

    def fit_to_view(self) -> None:
        if self._image_size == (0, 0):
            return
        iw, ih = self._image_size
        self.resetTransform()
        self.fitInView(QRectF(0, 0, iw, ih), Qt.KeepAspectRatio)
        self._zoom = max(self._min_zoom, min(self._max_zoom, self.transform().m11()))

    def paintEvent(self, event) -> None:
        """场景绘制完成后，在 viewport 上叠加屏幕层遮罩。

        模仿 moonrailgun 的做法：遮罩尺寸 = 图片的屏幕显示矩形，
        内部用 clip-path 风格的多边形挖空选区。这样：
        - 遮罩只覆盖图片区，不污染 viewport 边缘空白
        - 不管缩放比例如何都自然跟随
        """
        super().paintEvent(event)
        if self._crop_box is None or self._image_size == (0, 0) or self._pix_item is None:
            return

        from PySide6.QtGui import QPainter, QPolygonF
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
        painter.setPen(Qt.NoPen)

        # 图片的 4 个角在屏幕上的位置（=遮罩的外框）
        iw, ih = self._image_size
        img_tl = self.mapFromScene(0, 0)
        img_br = self.mapFromScene(iw, ih)
        ix0, iy0 = img_tl.x(), img_tl.y()
        ix1, iy1 = img_br.x(), img_br.y()
        vw = self.viewport().width()
        vh = self.viewport().height()

        # 选区在屏幕上的位置
        sel_tl = self.mapFromScene(self._sel_rect.topLeft())
        sel_br = self.mapFromScene(self._sel_rect.bottomRight())
        sx0, sy0 = sel_tl.x(), sel_tl.y()
        sx1, sy1 = sel_br.x(), sel_br.y()

        # 如果图片完全在 viewport 外，不画
        if ix1 < 0 or iy1 < 0 or ix0 > vw or iy0 > vh:
            return

        # 用一个外部矩形（覆盖整个 viewport）减去图片外框 → 仅画图片区域外的遮罩
        # 然后图片区域内部再画一个"框形"（挖空选区）
        # 简化：直接用多边形（顺时针绕图片外框 + 逆时针绕选区），这样中间就镂空
        poly = QPolygonF()
        # 外框顺时针
        poly.append(QPointF(max(0, ix0), max(0, iy0)))
        poly.append(QPointF(min(vw, ix1), max(0, iy0)))
        poly.append(QPointF(min(vw, ix1), min(vh, iy1)))
        poly.append(QPointF(max(0, ix0), min(vh, iy1)))
        # 内框逆时针（挖空选区）
        poly.append(QPointF(sx0, sy0))
        poly.append(QPointF(sx0, sy1))
        poly.append(QPointF(sx1, sy1))
        poly.append(QPointF(sx1, sy0))

        # Qt 的 QPainter 没法直接做偶奇规则，所以拆成两块绘制
        # 1) 图片外 → viewport 内（外侧遮罩）
        if ix0 > 0 or iy0 > 0 or ix1 < vw or iy1 < vh:
            painter.drawRect(0, 0, vw, max(0, iy0))
            painter.drawRect(0, min(vh, iy1), vw, max(0, vh - iy1))
            painter.drawRect(0, max(0, iy0), max(0, ix0), min(vh, iy1) - max(0, iy0))
            painter.drawRect(min(vw, ix1), max(0, iy0), max(0, vw - ix1), min(vh, iy1) - max(0, iy0))

        # 2) 图片内部挖空选区（画 4 个矩形包围选区）
        if sx1 > 0 and sx0 < vw and sy1 > 0 and sy0 < vh:
            ax0 = max(sx0, ix0)
            ay0 = max(sy0, iy0)
            ax1 = min(sx1, ix1)
            ay1 = min(sy1, iy1)
            # 上
            painter.drawRect(ax0, iy0, ax1 - ax0, ay0 - iy0)
            # 下
            painter.drawRect(ax0, ay1, ax1 - ax0, iy1 - ay1)
            # 左
            painter.drawRect(ix0, ay0, ax0 - ix0, ay1 - ay0)
            # 右
            painter.drawRect(ax1, ay0, ix1 - ax1, ay1 - ay0)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 1.2 if delta > 0 else 1.0 / 1.2
            new_zoom = self._zoom * factor
            new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
            actual = new_zoom / self._zoom
            self._zoom = new_zoom
            self.scale(actual, actual)
            event.accept()
        else:
            super().wheelEvent(event)

    def _map_to_image(self, pt: QPointF) -> QPointF:
        m = self.transform()
        return QPointF(pt.x() / m.m11(), pt.y() / m.m22())

    def _hit_handle(self, scene_pt: QPointF) -> str | None:
        """检测鼠标是否悬停在某手柄上。"""
        if self._crop_box is None:
            return None
        for name, h in self._crop_box._handles.items():
            hp = h.scenePos()
            if abs(scene_pt.x() - hp.x()) <= HANDLE_SIZE and abs(scene_pt.y() - hp.y()) <= HANDLE_SIZE:
                return name
        return None

    def _is_in_or_near_selection(self, img_pt: QPointF) -> bool:
        """移动模式用：扩大热区，把整框（含边框外 N 像素）都视为可拖。"""
        if self._sel_rect.isEmpty():
            return False
        pad = max(8.0, HANDLE_SIZE * 2)
        return self._sel_rect.adjusted(-pad, -pad, pad, pad).contains(img_pt)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pt = self.mapToScene(event.pos())
        img_pt = self._map_to_image(scene_pt)

        # 移动模式：禁用手柄拖拽和空白拉框，只能整体平移选区
        if self._mode == self.MODE_MOVE:
            # 整框（含边框）热区：即便鼠标落在边框/手柄附近也能拖
            if self._is_in_or_near_selection(img_pt):
                self._drag_mode = self.DRAG_MOVE
                self._drag_start = img_pt
                self._drag_rect_start = QRectF(self._sel_rect)
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
            else:
                # 框外点击不做任何事
                event.accept()
            return

        handle = self._hit_handle(scene_pt)
        if handle:
            self._drag_mode = self.DRAG_HANDLE
            self._drag_handle = handle
            self._drag_start = img_pt
            self._drag_rect_start = QRectF(self._sel_rect)
            self.setCursor(self._cursor_for_handle(handle))
            event.accept()
            return

        if self._sel_rect.contains(img_pt):
            self._drag_mode = self.DRAG_MOVE
            self._drag_start = img_pt
            self._drag_rect_start = QRectF(self._sel_rect)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # 点击空白 → 从该点拉出新选区
        self._drag_mode = self.DRAG_HANDLE
        self._drag_handle = "se"
        self._drag_start = img_pt
        self._drag_rect_start = QRectF(img_pt, img_pt)
        self.setCursor(Qt.CrossCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_mode == self.DRAG_NONE:
            if self._crop_box:
                if self._mode == self.MODE_MOVE:
                    # 移动模式：整框热区显示 OpenHandCursor，提示可拖
                    img_pt = self._map_to_image(self.mapToScene(event.pos()))
                    if self._is_in_or_near_selection(img_pt):
                        self.setCursor(Qt.OpenHandCursor)
                    else:
                        self.setCursor(Qt.ArrowCursor)
                else:
                    h = self._hit_handle(self.mapToScene(event.pos()))
                    self.setCursor(self._cursor_for_handle(h) if h else Qt.ArrowCursor)
            super().mouseMoveEvent(event)
            return

        scene_pt = self.mapToScene(event.pos())
        img_pt = self._map_to_image(scene_pt)
        dx = img_pt.x() - self._drag_start.x()
        dy = img_pt.y() - self._drag_start.y()
        iw, ih = self._image_size
        min_s = CropBox.MIN_SIZE

        r = QRectF(self._drag_rect_start)

        if self._drag_mode == self.DRAG_MOVE:
            nx = max(0, min(iw - r.width(),  r.x() + dx))
            ny = max(0, min(ih - r.height(), r.y() + dy))
            r.moveTo(nx, ny)

        elif self._drag_mode == self.DRAG_HANDLE:
            h = self._drag_handle
            x0, y0 = r.x(), r.y()
            x1, y1 = r.x() + r.width(), r.y() + r.height()

            if "w" in h:
                nx0 = max(0, min(x1 - min_s, x0 + dx))
                r.setLeft(nx0)
            if "e" in h:
                nx1 = max(min_s, min(iw, x1 + dx))
                r.setRight(nx1)
            if "n" in h:
                ny0 = max(0, min(y1 - min_s, y0 + dy))
                r.setTop(ny0)
            if "s" in h:
                ny1 = max(min_s, min(ih, y1 + dy))
                r.setBottom(ny1)

        r = r.normalized().intersected(QRectF(0, 0, iw, ih))
        self._sel_rect = r
        if self._crop_box:
            self._crop_box.setRect(r)
        # 立即请求重画 viewport（让 paintEvent 重绘屏幕层遮罩，否则旧遮罩会留下残影）
        self.viewport().update()
        self.selection_changed.emit(r.toRect())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_mode != self.DRAG_NONE:
            self.selection_changed.emit(self._sel_rect.toRect())
        self._drag_mode = self.DRAG_NONE
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    @staticmethod
    def _cursor_for_handle(h: str | None) -> Qt.CursorShape:
        cursors = {
            "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
            "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
            "n":  Qt.SizeVerCursor,   "s":  Qt.SizeVerCursor,
            "w":  Qt.SizeHorCursor,   "e":  Qt.SizeHorCursor,
        }
        return cursors.get(h or "", Qt.ArrowCursor)


# ---------------------------------------------------------------------------
# 结果预览视图
# ---------------------------------------------------------------------------

class PreviewView(QGraphicsView):
    """只读预览结果。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        # 棋盘格 + 半透明遮罩，方便看出图片边界
        self.setBackgroundBrush(_make_checker_brush())
        # 滚动视图仍保持智能重画，仅 viewport 已足够
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(300, 300)
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 32.0
        self._pix_item: QGraphicsPixmapItem | None = None
        self._border_item: QGraphicsRectItem | None = None
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._scene.clear()
        self._pix_item = None
        self._border_item = None
        PlaceholderText("(无预览)", self._scene, 85, 135)

    def load(self, arr: np.ndarray | None) -> None:
        self._scene.clear()
        self._pix_item = None
        self._border_item = None
        if arr is None:
            self._show_placeholder()
            return
        h, w = arr.shape[:2]
        self._pix_item = QGraphicsPixmapItem(pil_to_qpixmap(np_to_pil(arr)))
        self._pix_item.setZValue(0)
        self._scene.addItem(self._pix_item)

        # 1px 边界框，让用户看清图片范围
        border = QGraphicsRectItem(0, 0, w, h)
        border.setPen(_xp_border_pen())
        border.setBrush(QBrush(Qt.NoBrush))
        border.setZValue(0.5)
        self._scene.addItem(border)
        self._border_item = border

        # 加 padding 让 border 不贴住 viewport 边缘
        pad = 16
        self._scene.setSceneRect(-pad, -pad, w + 2 * pad, h + 2 * pad)
        self.fit_to_view()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 1.2 if delta > 0 else 1.0 / 1.2
            new_zoom = self._zoom * factor
            new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
            actual = new_zoom / self._zoom
            self._zoom = new_zoom
            self.scale(actual, actual)
            event.accept()
        else:
            super().wheelEvent(event)

    def fit_to_view(self) -> None:
        r = self._scene.sceneRect()
        if r.isEmpty():
            return
        self.resetTransform()
        self.fitInView(r, Qt.KeepAspectRatio)
        self._zoom = max(self._min_zoom, min(self._max_zoom, self.transform().m11()))


# ---------------------------------------------------------------------------
# 主控件
# ---------------------------------------------------------------------------

class ImageCropWidget(QWidget):
    """图片裁剪到指定分辨率。

    支持：
    - 在原图上拖拽选区（8个手柄 + 移动 + 空白拉框）
    - 通过左上角 / 居中 / 右下角锚点设置裁剪区域
    - 生成目标尺寸结果并导出
    """

    _buf: Optional[ImageBuffer] = None

    @staticmethod
    def set_buffer_ref(buf: Optional[ImageBuffer]) -> None:
        ImageCropWidget._buf = buf

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buf: Optional[ImageBuffer] = ImageCropWidget._buf
        self._source: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._src_w = 0
        self._src_h = 0
        self._src_tab = ""
        self._syncing = False  # 防止 spin ↔ selection_changed 死循环

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- 工具栏 ----
        tb = QHBoxLayout()
        tb.setSpacing(8)

        btn_open = QPushButton("打开图片…")
        btn_open.clicked.connect(self._on_open)
        tb.addWidget(btn_open)

        btn_load = QPushButton("从暂存区载入")
        btn_load.clicked.connect(self._on_load_buffer)
        tb.addWidget(btn_load)

        tb.addSpacing(16)
        tb.addWidget(QLabel("模式:"))
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(4)
        for idx, (label, mode, tip) in enumerate([
            ("调整大小", CropView.MODE_RESIZE,
             "调整大小：可拖拽 8 个手柄或在空白处拉出新框"),
            ("移动位置", CropView.MODE_MOVE,
             "移动位置：固定当前框大小，只能整体平移"),
        ]):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(idx == 0)
            btn.setToolTip(tip)
            btn.mode_value = mode  # type: ignore[attr-defined]
            self._mode_group.addButton(btn, idx)
            mode_layout.addWidget(btn)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        tb.addLayout(mode_layout)

        tb.addSpacing(16)
        tb.addWidget(QLabel("W:"))
        self.spin_tw = QSpinBox()
        self.spin_tw.setRange(1, 8192)
        self.spin_tw.setValue(48)
        self.spin_tw.setFixedWidth(75)
        self.spin_tw.setToolTip("裁剪框当前宽度（也可在此修改框大小）")
        self.spin_tw.valueChanged.connect(self._on_target_changed)
        tb.addWidget(self.spin_tw)

        tb.addWidget(QLabel("H:"))
        self.spin_th = QSpinBox()
        self.spin_th.setRange(1, 8192)
        self.spin_th.setValue(48)
        self.spin_th.setFixedWidth(75)
        self.spin_th.setToolTip("裁剪框当前高度（也可在此修改框大小）")
        self.spin_th.valueChanged.connect(self._on_target_changed)
        tb.addWidget(self.spin_th)

        tb.addSpacing(16)
        tb.addWidget(QLabel("锚点:"))
        self._anchor_group = QButtonGroup(self)
        self._anchor_group.setExclusive(True)
        anchor_layout = QHBoxLayout()
        anchor_layout.setSpacing(4)
        for idx, label in enumerate(["左上", "居中", "右下"]):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(idx == 1)
            self._anchor_group.addButton(btn, idx)
            anchor_layout.addWidget(btn)
        self._anchor_group.idClicked.connect(self._on_anchor_changed)
        tb.addLayout(anchor_layout)

        tb.addSpacing(16)
        self.btn_apply = QPushButton("生成裁剪预览")
        self.btn_apply.setStyleSheet("font-weight: bold;")
        self.btn_apply.clicked.connect(self._on_apply)
        tb.addWidget(self.btn_apply)

        tb.addStretch(1)
        root.addLayout(tb)

        # ---- 主视图 ----
        body = QHBoxLayout()
        body.setSpacing(12)

        src_wrap = QVBoxLayout()
        src_wrap.setSpacing(4)
        hdr = QLabel("原图（拖拽选区或手柄调整）")
        hdr.setStyleSheet("font-weight: bold; font-size: 13px;")
        src_wrap.addWidget(hdr)
        self.crop_view = CropView()
        self.crop_view.selection_changed.connect(self._on_selection_changed)
        # 每次选区变化（拖动/手柄）实时刷新预览
        self.crop_view.selection_changed.connect(lambda _: self._update_preview())
        src_wrap.addWidget(self.crop_view, 1)
        self.lbl_info = QLabel("尚未载入图片")
        self.lbl_info.setStyleSheet("color: #888; font-size: 11px;")
        src_wrap.addWidget(self.lbl_info)
        body.addLayout(src_wrap, 2)

        prev_wrap = QVBoxLayout()
        prev_wrap.setSpacing(4)
        hdr2 = QLabel("裁剪结果预览")
        hdr2.setStyleSheet("font-weight: bold; font-size: 13px;")
        prev_wrap.addWidget(hdr2)
        self.preview_view = PreviewView()
        prev_wrap.addWidget(self.preview_view, 1)
        self.lbl_result = QLabel("")
        self.lbl_result.setStyleSheet("color: #888; font-size: 11px;")
        prev_wrap.addWidget(self.lbl_result)
        body.addLayout(prev_wrap, 1)

        root.addLayout(body, 1)

        # ---- 底部动作 ----
        action = QHBoxLayout()
        action.setSpacing(8)
        self.btn_export = QPushButton("导出 PNG")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        action.addWidget(self.btn_export)
        self.btn_to_buf = QPushButton("加入暂存区")
        self.btn_to_buf.setEnabled(False)
        self.btn_to_buf.clicked.connect(self._on_to_buffer)
        action.addWidget(self.btn_to_buf)
        action.addStretch(1)
        self.lbl_status = QLabel("打开或从暂存区载入图片，拖拽选区。W/H 实时显示框尺寸，也可直接输入。")
        self.lbl_status.setStyleSheet("color: #888; font-size: 11px;")
        action.addWidget(self.lbl_status)
        root.addLayout(action)

    # ------------------------------------------------------------------
    # 载入
    # ------------------------------------------------------------------
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "",
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        try:
            pil_img = Image.open(path).convert("RGBA")
            arr = np.array(pil_img, dtype=np.uint8)
        except Exception as exc:
            QMessageBox.critical(self, "打开失败", f"无法读取图片：\n{exc}")
            return
        name = path.split("/")[-1]
        self._load(arr, f"裁剪: {name}")

    def _on_load_buffer(self) -> None:
        if self._buf is None:
            QMessageBox.warning(self, "暂存区不可用", "暂存区未初始化")
            return
        items = self._buf.items()
        if not items:
            QMessageBox.information(self, "暂存区为空", "暂存区里没有图片")
            return
        latest = items[-1]
        self._load(latest["image"], latest.get("source_tab", ""))

    def _load(self, rgba: np.ndarray, source_tab: str) -> None:
        if rgba.ndim == 2:
            rgba = np.stack([rgba] * 3, axis=-1)
        if rgba.shape[2] == 3:
            rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, dtype=np.uint8)])
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)

        self._source = rgba
        self._src_tab = source_tab
        self._src_h, self._src_w = rgba.shape[:2]

        self.crop_view.load_image(rgba)
        self.lbl_info.setText(f"{self._src_w} × {self._src_h} px")

        tw = self.spin_tw.value()
        th = self.spin_th.value()
        self._set_selection_by_anchor(tw, th, self._anchor_group.checkedId())
        self._result = None
        self.preview_view.load(None)
        self.btn_export.setEnabled(False)
        self.btn_to_buf.setEnabled(False)
        self.lbl_result.setText("")
        self.lbl_status.setText(f"已载入 {self._src_w}×{self._src_h} | 目标 {tw}×{th} | 拖拽选区或点「生成裁剪预览」")

    # ------------------------------------------------------------------
    # 选区 / 锚点 / 目标尺寸变化
    # ------------------------------------------------------------------
    def _on_mode_changed(self, mode_id: int) -> None:
        """工具栏模式按钮切换。"""
        btn = self._mode_group.button(mode_id)
        if btn is None:
            return
        mode = getattr(btn, "mode_value", CropView.MODE_RESIZE)
        self.crop_view.set_mode(mode)

    def _on_selection_changed(self, rect: QRect) -> None:
        """选区被拖动/手柄调整后实时刷新 W/H 显示。"""
        if self._syncing:
            return
        self._syncing = True
        try:
            self.spin_tw.blockSignals(True)
            self.spin_th.blockSignals(True)
            self.spin_tw.setValue(max(1, rect.width()))
            self.spin_th.setValue(max(1, rect.height()))
        finally:
            self.spin_tw.blockSignals(False)
            self.spin_th.blockSignals(False)
            self._syncing = False

    def _on_target_changed(self) -> None:
        """用户在 W/H spinbox 里改值 → 调整当前选区大小（左上角为基准），并即时刷新预览。"""
        if self._syncing or self._source is None:
            return
        tw = max(1, self.spin_tw.value())
        th = max(1, self.spin_th.value())
        sel = self.crop_view.selection_rect()
        x = sel.x() if not sel.isEmpty() else 0
        y = sel.y() if not sel.isEmpty() else 0
        tw = max(1, min(tw, self._src_w - x))
        th = max(1, min(th, self._src_h - y))
        self._syncing = True
        try:
            self.crop_view.set_selection(QRect(x, y, tw, th))
        finally:
            self._syncing = False
        self._update_preview()

    def _reposition_selection(self, anchor_id: int) -> None:
        tw = self.spin_tw.value()
        th = self.spin_th.value()
        anchor = self._anchor_group.checkedId()
        self._set_selection_by_anchor(tw, th, anchor_id if anchor_id is not None else anchor)

    def _on_anchor_changed(self, anchor_id: int) -> None:
        self._reposition_selection(anchor_id)

    def _on_target_changed(self) -> None:
        tw = self.spin_tw.value()
        th = self.spin_th.value()
        anchor = self._anchor_group.checkedId()
        self._set_selection_by_anchor(tw, th, anchor)

    def _reposition_selection(self, anchor_id: int) -> None:
        tw = self.spin_tw.value()
        th = self.spin_th.value()
        self._set_selection_by_anchor(tw, th, anchor_id)

    def _set_selection_by_anchor(self, tw: int, th: int, anchor_id: int) -> None:
        if self._source is None:
            return
        if anchor_id == 0:
            x, y = 0, 0
        elif anchor_id == 2:
            x = max(0, self._src_w - tw)
            y = max(0, self._src_h - th)
        else:
            x = max(0, (self._src_w - tw) // 2)
            y = max(0, (self._src_h - th) // 2)
        x = max(0, min(self._src_w - 1, x))
        y = max(0, min(self._src_h - 1, y))
        w = min(tw, self._src_w - x)
        h = min(th, self._src_h - y)
        self.crop_view.set_selection(QRect(x, y, w, h))

    # ------------------------------------------------------------------
    # 生成裁剪结果
    # ------------------------------------------------------------------
    def _on_apply(self) -> None:
        if self._source is None:
            QMessageBox.information(self, "没有图片", "请先打开或载入图片")
            return

        sel = self.crop_view.selection_rect().normalized()
        if sel.isEmpty():
            QMessageBox.warning(self, "选区无效", "裁剪选区为空，请先拖拽选择区域")
            return

        x, y, w, h = sel.x(), sel.y(), sel.width(), sel.height()
        cropped = self._source[y : y + h, x : x + w].copy()
        out_w, out_h = w, h

        self._result = np.ascontiguousarray(cropped, dtype=np.uint8)
        self.preview_view.load(self._result)

        size_text = f"{out_w}×{out_h}"

        self.lbl_result.setText(f"裁剪: ({x},{y}) {w}×{h} → {size_text}")
        self.btn_export.setEnabled(True)

    def _update_preview(self) -> None:
        """实时刷新预览（不弹错误对话框；选区/图片为空就直接清空）。"""
        if self._source is None:
            return
        sel = self.crop_view.selection_rect().normalized()
        if sel.isEmpty():
            return
        x, y, w, h = sel.x(), sel.y(), sel.width(), sel.height()
        if w < 1 or h < 1:
            return
        if x >= self._src_w or y >= self._src_h:
            return
        w = min(w, self._src_w - x)
        h = min(h, self._src_h - y)
        cropped = self._source[y : y + h, x : x + w].copy()
        out_w, out_h = w, h
        self._result = np.ascontiguousarray(cropped, dtype=np.uint8)
        self.preview_view.load(self._result)
        size_text = f"{out_w}×{out_h}"
        self.lbl_result.setText(f"裁剪: ({x},{y}) {w}×{h} → {size_text}")
        self.btn_export.setEnabled(True)
        self.btn_to_buf.setEnabled(True)
        self.lbl_status.setText(f"裁剪完成：({x},{y}) {w}×{h} → {size_text}")

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def _on_export(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出图片", "",
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if not path:
            return
        try:
            pil = Image.fromarray(self._result, mode="RGBA")
            if path.lower().endswith(".jpg"):
                rgb = np.array(pil.convert("RGB"))
                Image.fromarray(rgb).save(path, "JPEG", quality=95)
            else:
                pil.save(path, "PNG")
            self.lbl_status.setText(f"已导出到: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _on_to_buffer(self) -> None:
        if self._result is None:
            return
        if self._buf is None:
            QMessageBox.warning(self, "暂存区不可用", "暂存区未初始化")
            return
        tag = f"裁剪 {self._result.shape[1]}×{self._result.shape[0]}"
        self._buf.push(self._result, source_tab=self._src_tab or tag, source_file=tag)
        self.lbl_status.setText(f"已加入暂存区：{tag}")
        QMessageBox.information(self, "完成", f"已把结果图加入暂存区。\n{tag}")
