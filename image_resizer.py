"""
图像缩放 / 裁剪模块：把一张图缩放到指定尺寸，或裁剪出指定区域。

用法：
    from image_resizer import ImageResizerWidget
    ImageResizerWidget.set_buffer_ref(buf)
    tabs.addTab(ImageResizerWidget(), "📐 缩放裁剪")
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt, QRect, QRectF, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
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

from perfect_pixel.app_core.image_buffer import ImageBuffer


# ---------------------------------------------------------------------------
# PIL -> QPixmap（最可靠的方式，走 Qt 自己的转换逻辑）
# ---------------------------------------------------------------------------

def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """PIL Image -> QPixmap，用 Qt 自己的 ImageQt 转换，无字节序问题。"""
    return QPixmap.fromImage(ImageQt(img))


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy HxWx3/4 -> PIL RGBA Image。"""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 3:
        arr = np.dstack([arr, np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)])
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


# ---------------------------------------------------------------------------
# 带网格的原图视图（支持 Ctrl+滚轮缩放、拖动平移）
# ---------------------------------------------------------------------------

class SourceImageView(QGraphicsView):
    """显示原图，Ctrl+滚轮缩放，拖动平移。

    缩放 > 4× 时自动叠加像素网格，帮助对齐像素边界。
    """

    selection_changed = Signal(QRect)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(360, 300)

        self._pix_item: QGraphicsPixmapItem | None = None
        self._grid_item: QGraphicsRectItem | None = None
        self._sel_item: QGraphicsRectItem | None = None
        self._sel_rect = QRect()
        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 32.0

        self._ph_label: QLabel = QLabel("(无图片)")
        self._ph_label.setAlignment(Qt.AlignCenter)
        self._ph_label.setStyleSheet("color: #555; font-size: 14px;")
        self._ph_label.setFixedSize(360, 300)
        self._scene.addWidget(self._ph_label)

    def load_image(self, arr: np.ndarray) -> None:
        """载入一张 RGBA numpy 数组。"""
        self._ph_label.hide()
        h, w = arr.shape[:2]

        pix = pil_to_qpixmap(np_to_pil(arr))

        if self._pix_item:
            self._scene.removeItem(self._pix_item)
        self._pix_item = QGraphicsPixmapItem(pix)
        self._pix_item.setZValue(0)
        self._scene.addItem(self._pix_item)

        self._scene.setSceneRect(0, 0, w, h)
        self._sel_rect = QRect(0, 0, w, h)
        self._update_selection_overlay()
        self._update_grid()
        self.fit_to_view()

    def clear(self) -> None:
        self._scene.clear()
        self._pix_item = None
        self._grid_item = None
        self._sel_item = None
        self._ph_label = QLabel("(无图片)")
        self._ph_label.setAlignment(Qt.AlignCenter)
        self._ph_label.setStyleSheet("color: #555; font-size: 14px;")
        self._ph_label.setFixedSize(360, 300)
        self._scene.addWidget(self._ph_label)

    def set_selection(self, rect: QRect) -> None:
        """设置裁剪选区（图像坐标）。"""
        self._sel_rect = QRect(rect)
        self._update_selection_overlay()

    def selection_rect(self) -> QRect:
        return QRect(self._sel_rect)

    def fit_to_view(self) -> None:
        r = self._scene.sceneRect()
        if r.isEmpty():
            return
        self.resetTransform()
        self.fitInView(r, Qt.KeepAspectRatio)
        self._zoom = max(self._min_zoom, min(self._max_zoom, self.transform().m11()))
        self._update_grid()

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
            self._update_grid()
            event.accept()
        else:
            super().wheelEvent(event)

    def _update_grid(self) -> None:
        """放大 > 4× 时叠加像素网格。"""
        show_grid = self._zoom >= 4.0
        if self._grid_item:
            self._grid_item.setVisible(show_grid)

    def _update_selection_overlay(self) -> None:
        r = self._sel_rect
        if not self._sel_item:
            self._sel_item = QGraphicsRectItem()
            self._sel_item.setZValue(10)
            self._scene.addItem(self._sel_item)
        self._sel_item.setRect(QRectF(r))
        self._sel_item.setBrush(QBrush(QColor(0, 120, 255, 50)))
        pen = QPen(QColor(0, 180, 255), 1.5 / max(self._zoom, 1.0))
        pen.setCosmetic(True)
        self._sel_item.setPen(pen)


# ---------------------------------------------------------------------------
# 结果预览视图
# ---------------------------------------------------------------------------

class ResultView(QGraphicsView):
    """只读显示缩放/裁剪结果的视图，支持 Ctrl+滚轮缩放、拖动平移。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(360, 300)
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 32.0
        self._ph_label: QLabel = QLabel("(无预览)")
        self._ph_label.setAlignment(Qt.AlignCenter)
        self._ph_label.setStyleSheet("color: #555; font-size: 14px;")
        self._ph_label.setFixedSize(360, 300)
        self._scene.addWidget(self._ph_label)

    def load_result(self, arr: np.ndarray | None) -> None:
        """显示一张 RGBA numpy 数组作为结果。"""
        self._ph_label.hide()
        self._scene.clear()
        if arr is None:
            self._ph_label = QLabel("(无预览)")
            self._ph_label.setAlignment(Qt.AlignCenter)
            self._ph_label.setStyleSheet("color: #555; font-size: 14px;")
            self._ph_label.setFixedSize(360, 300)
            self._scene.addWidget(self._ph_label)
            return
        h, w = arr.shape[:2]
        item = QGraphicsPixmapItem(pil_to_qpixmap(np_to_pil(arr)))
        self._scene.addItem(item)
        self._scene.setSceneRect(0, 0, w, h)
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

class ImageResizerWidget(QWidget):
    """图像缩放 / 裁剪：
    - 缩放：指定目标宽 × 高，使用最近邻（像素画）或双线性插值
    - 裁剪：从原图截取 X,Y,W,H 区域
    支持从暂存区载入、导出 PNG、加回暂存区。
    """

    _buf: Optional[ImageBuffer] = None

    @staticmethod
    def set_buffer_ref(buf: Optional[ImageBuffer]) -> None:
        ImageResizerWidget._buf = buf

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buf: Optional[ImageBuffer] = ImageResizerWidget._buf
        self._source: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._source_tab: str = ""
        self._src_h = 0
        self._src_w = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        btn_open = QPushButton("打开图片…")
        btn_open.clicked.connect(self._on_open)
        toolbar.addWidget(btn_open)

        btn_load_buf = QPushButton("从暂存区载入")
        btn_load_buf.clicked.connect(self._on_load_from_buffer)
        toolbar.addWidget(btn_load_buf)

        toolbar.addSpacing(16)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        mode_row.addWidget(QLabel("模式:"))
        self.btn_mode_resize = QPushButton("缩放")
        self.btn_mode_resize.setCheckable(True)
        self.btn_mode_resize.setChecked(True)
        self._mode_group.addButton(self.btn_mode_resize, 0)
        self.btn_mode_crop = QPushButton("裁剪")
        self.btn_mode_crop.setCheckable(True)
        self._mode_group.addButton(self.btn_mode_crop, 1)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        mode_row.addWidget(self.btn_mode_resize)
        mode_row.addWidget(self.btn_mode_crop)
        toolbar.addLayout(mode_row)

        toolbar.addSpacing(16)

        self._resize_panel = QWidget()
        rp = QHBoxLayout(self._resize_panel)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(6)
        rp.addWidget(QLabel("宽:"))
        self.spin_new_w = QSpinBox()
        self.spin_new_w.setRange(1, 8192)
        self.spin_new_w.setValue(24)
        self.spin_new_w.setFixedWidth(80)
        self.spin_new_w.valueChanged.connect(self._on_params_changed)
        rp.addWidget(self.spin_new_w)
        rp.addWidget(QLabel("高:"))
        self.spin_new_h = QSpinBox()
        self.spin_new_h.setRange(1, 8192)
        self.spin_new_h.setValue(24)
        self.spin_new_h.setFixedWidth(80)
        self.spin_new_h.valueChanged.connect(self._on_params_changed)
        rp.addWidget(self.spin_new_h)
        rp.addSpacing(8)
        rp.addWidget(QLabel("插值:"))
        self.combo_interp = QComboBox()
        self.combo_interp.addItems(["最近邻（像素画）", "双线性（普通图）", "双三次（高质量）"])
        self.combo_interp.setFixedWidth(140)
        self.combo_interp.currentIndexChanged.connect(self._on_params_changed)
        rp.addWidget(self.combo_interp)
        toolbar.addWidget(self._resize_panel)

        self._crop_panel = QWidget()
        self._crop_panel.setVisible(False)
        cp = QHBoxLayout(self._crop_panel)
        cp.setContentsMargins(0, 0, 0, 0)
        cp.setSpacing(6)
        cp.addWidget(QLabel("X:"))
        self.spin_crop_x = QSpinBox()
        self.spin_crop_x.setRange(0, 8192)
        self.spin_crop_x.setValue(0)
        self.spin_crop_x.setFixedWidth(80)
        self.spin_crop_x.valueChanged.connect(self._on_params_changed)
        cp.addWidget(self.spin_crop_x)
        cp.addWidget(QLabel("Y:"))
        self.spin_crop_y = QSpinBox()
        self.spin_crop_y.setRange(0, 8192)
        self.spin_crop_y.setValue(0)
        self.spin_crop_y.setFixedWidth(80)
        self.spin_crop_y.valueChanged.connect(self._on_params_changed)
        cp.addWidget(self.spin_crop_y)
        cp.addWidget(QLabel("W:"))
        self.spin_crop_w = QSpinBox()
        self.spin_crop_w.setRange(1, 8192)
        self.spin_crop_w.setValue(24)
        self.spin_crop_w.setFixedWidth(80)
        self.spin_crop_w.valueChanged.connect(self._on_params_changed)
        cp.addWidget(self.spin_crop_w)
        cp.addWidget(QLabel("H:"))
        self.spin_crop_h = QSpinBox()
        self.spin_crop_h.setRange(1, 8192)
        self.spin_crop_h.setValue(24)
        self.spin_crop_h.setFixedWidth(80)
        self.spin_crop_h.valueChanged.connect(self._on_params_changed)
        cp.addWidget(self.spin_crop_h)
        cp.addSpacing(8)
        self.btn_fit_sel = QPushButton("选区适应窗口")
        self.btn_fit_sel.clicked.connect(self._on_fit_selection)
        cp.addWidget(self.btn_fit_sel)
        toolbar.addWidget(self._crop_panel)

        toolbar.addStretch(1)
        self.btn_apply = QPushButton("生成预览")
        self.btn_apply.setStyleSheet("font-weight: bold;")
        self.btn_apply.clicked.connect(self._on_apply)
        toolbar.addWidget(self.btn_apply)

        root.addLayout(toolbar)

        body = QHBoxLayout()
        body.setSpacing(12)

        src_wrap = QVBoxLayout()
        src_wrap.setSpacing(4)
        hdr_src = QLabel("原图")
        hdr_src.setStyleSheet("font-weight: bold; font-size: 13px;")
        src_wrap.addWidget(hdr_src)
        self.src_view = SourceImageView()
        self.src_view.selection_changed.connect(self._on_selection_changed)
        src_wrap.addWidget(self.src_view, 1)
        self.lbl_src_info = QLabel("尚未载入图片")
        self.lbl_src_info.setStyleSheet("color: #888; font-size: 11px;")
        src_wrap.addWidget(self.lbl_src_info)
        body.addLayout(src_wrap, 1)

        res_wrap = QVBoxLayout()
        res_wrap.setSpacing(4)
        hdr_res = QLabel("结果预览")
        hdr_res.setStyleSheet("font-weight: bold; font-size: 13px;")
        res_wrap.addWidget(hdr_res)
        self.result_view = ResultView()
        res_wrap.addWidget(self.result_view, 1)
        self.lbl_result_info = QLabel("")
        self.lbl_result_info.setStyleSheet("color: #888; font-size: 11px;")
        res_wrap.addWidget(self.lbl_result_info)
        body.addLayout(res_wrap, 1)

        root.addLayout(body, 1)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.btn_export = QPushButton("导出 PNG")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        action_row.addWidget(self.btn_export)
        self.btn_to_buffer = QPushButton("加入暂存区")
        self.btn_to_buffer.setEnabled(False)
        self.btn_to_buffer.clicked.connect(self._on_to_buffer)
        action_row.addWidget(self.btn_to_buffer)
        action_row.addStretch(1)
        self.lbl_status = QLabel("打开或从暂存区载入一张图片，再选择缩放或裁剪参数。")
        self.lbl_status.setStyleSheet("color: #888; font-size: 11px;")
        action_row.addWidget(self.lbl_status)
        root.addLayout(action_row)

    def _on_mode_changed(self, mode_id: int) -> None:
        if mode_id == 0:
            self._resize_panel.setVisible(True)
            self._crop_panel.setVisible(False)
            self._mode = "resize"
        else:
            self._resize_panel.setVisible(False)
            self._crop_panel.setVisible(True)
            self._mode = "crop"
        self._on_params_changed()

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
        self.load(arr, source_tab=f"缩放裁剪: {path.split('/')[-1]}")

    def _on_load_from_buffer(self) -> None:
        if self._buf is None:
            QMessageBox.warning(self, "暂存区不可用", "暂存区未初始化")
            return
        latest = self._buf.items()[-1] if self._buf.items() else None
        if latest is None:
            QMessageBox.information(self, "暂存区为空", "暂存区里没有图片")
            return
        self.load(latest["image"], source_tab=latest.get("source_tab", ""))

    def load(self, rgba: np.ndarray, source_tab: str = "") -> None:
        if rgba.ndim == 2:
            rgba = np.stack([rgba] * 3, axis=-1)
        if rgba.shape[2] == 3:
            rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, dtype=np.uint8)])
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)

        self._source = rgba
        self._source_tab = source_tab
        self._src_h, self._src_w = rgba.shape[:2]

        self.src_view.load_image(rgba)
        self.lbl_src_info.setText(f"{self._src_w} × {self._src_h} px")

        new_w = max(1, self._src_w // 2)
        new_h = max(1, self._src_h // 2)
        self.spin_new_w.setValue(new_w)
        self.spin_new_h.setValue(new_h)
        self.spin_new_w.setMaximum(self._src_w * 8)
        self.spin_new_h.setMaximum(self._src_h * 8)

        self.spin_crop_x.setRange(0, self._src_w - 1)
        self.spin_crop_y.setRange(0, self._src_h - 1)
        self.spin_crop_w.setRange(1, self._src_w)
        self.spin_crop_h.setRange(1, self._src_h)
        self.spin_crop_w.setValue(self._src_w)
        self.spin_crop_h.setValue(self._src_h)
        self.src_view.set_selection(QRect(0, 0, self._src_w, self._src_h))
        self._update_crop_spinboxes_from_selection()

        self._result = None
        self.result_view.load_result(None)
        self.btn_export.setEnabled(False)
        self.btn_to_buffer.setEnabled(False)
        self.lbl_result_info.setText("")
        self.lbl_status.setText(
            f"已载入: {self._src_w}×{self._src_h} | {source_tab or '直接打开'} | "
            f"点「生成预览」查看结果"
        )

    def _on_params_changed(self) -> None:
        if getattr(self, "_mode", "resize") != "crop":
            return
        self._update_selection_from_crop_spinboxes()

    def _update_selection_from_crop_spinboxes(self) -> None:
        x = self.spin_crop_x.value()
        y = self.spin_crop_y.value()
        w = self.spin_crop_w.value()
        h = self.spin_crop_h.value()
        x = min(x, max(0, self._src_w - 1))
        y = min(y, max(0, self._src_h - 1))
        w = min(w, self._src_w - x)
        h = min(h, self._src_h - y)
        self.src_view.set_selection(QRect(x, y, w, h))

    def _update_crop_spinboxes_from_selection(self) -> None:
        r = self.src_view.selection_rect()
        self.spin_crop_x.setValue(r.x())
        self.spin_crop_y.setValue(r.y())
        self.spin_crop_w.setValue(r.width())
        self.spin_crop_h.setValue(r.height())

    def _on_selection_changed(self, rect: QRect) -> None:
        self.spin_crop_x.setValue(rect.x())
        self.spin_crop_y.setValue(rect.y())
        self.spin_crop_w.setValue(rect.width())
        self.spin_crop_h.setValue(rect.height())

    def _on_fit_selection(self) -> None:
        r = self.src_view.selection_rect()
        if r.isEmpty():
            return
        self.src_view.fit_to_view()
        factor = max(100.0 / r.width(), 100.0 / r.height(), 1.0)
        self.src_view.scale(factor, factor)

    def _on_apply(self) -> None:
        if self._source is None:
            QMessageBox.information(self, "没有图片", "请先打开或载入一张图片。")
            return
        mode = getattr(self, "_mode", "resize")
        if mode == "resize":
            self._apply_resize()
        else:
            self._apply_crop()

    def _apply_resize(self) -> None:
        new_w = self.spin_new_w.value()
        new_h = self.spin_new_h.value()
        interp_idx = self.combo_interp.currentIndex()
        interp_map = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
        interp = interp_map[interp_idx]

        resized = cv2.resize(self._source, (new_w, new_h), interpolation=interp)
        self._result = np.ascontiguousarray(resized, dtype=np.uint8)
        self.result_view.load_result(self._result)
        interp_names = ["最近邻", "双线性", "双三次"]
        self.lbl_result_info.setText(
            f"{new_w} × {new_h} px  |  {interp_names[interp_idx]}  |  "
            f"原图 {self._src_w}×{self._src_h} → 结果"
        )
        self.btn_export.setEnabled(True)
        self.btn_to_buffer.setEnabled(True)
        self.lbl_status.setText(
            f"缩放完成：{self._src_w}×{self._src_h} → {new_w}×{new_h}（{interp_names[interp_idx]}）"
        )

    def _apply_crop(self) -> None:
        x = self.spin_crop_x.value()
        y = self.spin_crop_y.value()
        w = self.spin_crop_w.value()
        h = self.spin_crop_h.value()
        x = max(0, min(x, self._src_w - 1))
        y = max(0, min(y, self._src_h - 1))
        w = max(1, min(w, self._src_w - x))
        h = max(1, min(h, self._src_h - y))

        cropped = self._source[y : y + h, x : x + w].copy()
        self._result = np.ascontiguousarray(cropped, dtype=np.uint8)
        self.result_view.load_result(self._result)
        self.lbl_result_info.setText(
            f"裁剪区域: ({x},{y}) {w}×{h}  →  结果 {self._result.shape[1]}×{self._result.shape[0]}"
        )
        self.btn_export.setEnabled(True)
        self.btn_to_buffer.setEnabled(True)
        self.lbl_status.setText(
            f"裁剪完成：({x},{y}) {w}×{h} → 结果 {self._result.shape[1]}×{self._result.shape[0]}"
        )

    def _on_export(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出图片", "", "PNG (*.png);;JPEG (*.jpg)"
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
        mode = getattr(self, "_mode", "resize")
        tag = (f"缩放 {self._src_w}×{self._src_h}→{self._result.shape[1]}×{self._result.shape[0]}"
               if mode == "resize"
               else f"裁剪 ({self.spin_crop_x.value()},{self.spin_crop_y.value()}) {self._result.shape[1]}×{self._result.shape[0]}")
        self._buf.push(self._result, source_tab=self._source_tab or tag, source_file=tag)
        self.lbl_status.setText(f"已加入暂存区：{tag}")
        QMessageBox.information(self, "完成", f"已把结果图加入右侧暂存区。\n{tag}")
