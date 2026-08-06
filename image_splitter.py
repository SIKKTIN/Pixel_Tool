"""
图像切割模块：把一张图切成若干子图（网格切割 / 行列切分）。

用法：
    from image_splitter import ImageSplitterWidget
    ImageSplitterWidget.set_buffer_ref(buf)
    tabs.addTab(ImageSplitterWidget(), "✂️ 图像切割")
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QSize, QRectF
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QBrush,
    QWheelEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from desktop_app import ImageView, ImageBuffer, image_buffer, numpy_to_qpixmap


# ---------------------------------------------------------------------------
# 子图预览缩略图（带编号）
# ---------------------------------------------------------------------------
class TilePreview(QFrame):
    """显示一张切出来的子图，附带索引标签。"""

    def __init__(self, arr: np.ndarray, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._arr = arr
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #252525; border: 1px solid #444; border-radius: 6px; }"
        )
        self.setFixedSize(180, 200)

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        # 图片区
        img_label = QLabel()
        img_label.setFixedSize(168, 150)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("background: #2a2a2a; border: none;")
        if arr is not None and arr.size > 0:
            pix = numpy_to_qpixmap(arr)
            img_label.setPixmap(
                pix.scaled(168, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        v.addWidget(img_label)

        # 索引 + 尺寸
        info = QLabel(f"{label}   {arr.shape[1]} × {arr.shape[0]}")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #ddd; font-size: 11px; border: none;")
        v.addWidget(info)


# ---------------------------------------------------------------------------
# 缩放预览视图（基于 QGraphicsView，支持 Ctrl+滚轮缩放、拖动平移）
# ---------------------------------------------------------------------------
class PreviewView(QGraphicsView):
    """切割预览：内部用 QGraphicsScene 摆放子图 pixmap。
    支持 Ctrl+滚轮缩放、拖动平移、双击自适应全屏。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QBrush(QColor(35, 35, 35)))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(420, 320)
        self._zoom: float = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 8.0

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 1.25 if delta > 0 else 1.0 / 1.25
            new_zoom = self._zoom * factor
            new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
            actual_factor = new_zoom / self._zoom
            self._zoom = new_zoom
            self.scale(actual_factor, actual_factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """双击空白或子图：自适应整个场景到视口。"""
        if event.button() == Qt.LeftButton:
            self.fit_to_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def fit_to_view(self) -> None:
        """让 sceneRect 完整显示在视口内，并同步内部 _zoom。"""
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self.resetTransform()
        self.fitInView(rect, Qt.KeepAspectRatio)
        self._zoom = self.transform().m11()


# ---------------------------------------------------------------------------
# 切割主控件
# ---------------------------------------------------------------------------
class ImageSplitterWidget(QWidget):
    """打开 / 从暂存区载入图片，按行 × 列切成若干子图，预览并加入暂存区。"""

    _buf: Optional[ImageBuffer] = None

    @staticmethod
    def set_buffer_ref(buf: Optional[ImageBuffer]) -> None:
        ImageSplitterWidget._buf = buf

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buf: Optional[ImageBuffer] = ImageSplitterWidget._buf
        self._source_arr: Optional[np.ndarray] = None   # 当前原图 RGBA
        self._source_tab: str = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- 顶部操作栏 ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        btn_open = QPushButton("打开图片…")
        btn_open.clicked.connect(self._on_open)
        toolbar.addWidget(btn_open)

        btn_load_buf = QPushButton("从暂存区载入")
        btn_load_buf.clicked.connect(self._on_load_from_buffer)
        toolbar.addWidget(btn_load_buf)

        toolbar.addSpacing(20)

        toolbar.addWidget(QLabel("行数:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 20)
        self.spin_rows.setValue(2)
        toolbar.addWidget(self.spin_rows)

        toolbar.addWidget(QLabel("列数:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 20)
        self.spin_cols.setValue(2)
        toolbar.addWidget(self.spin_cols)

        toolbar.addStretch(1)

        btn_cut = QPushButton("执行切割")
        btn_cut.setStyleSheet("font-weight: bold;")
        btn_cut.clicked.connect(self._on_cut)
        toolbar.addWidget(btn_cut)

        toolbar.addSpacing(16)

        self.btn_fit = QPushButton("适应窗口")
        self.btn_fit.setToolTip("把预览自适应到当前窗口大小（也可双击预览区）")
        self.btn_fit.clicked.connect(self._on_fit_view)
        toolbar.addWidget(self.btn_fit)

        self.btn_push_selected = QPushButton("把选中的加入暂存区")
        self.btn_push_selected.setToolTip("把预览里当前选中的子图加入右侧暂存区")
        self.btn_push_selected.clicked.connect(self._on_push_selected)
        toolbar.addWidget(self.btn_push_selected)

        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self._on_select_all)
        toolbar.addWidget(self.btn_select_all)

        self.btn_clear_sel = QPushButton("清空选择")
        self.btn_clear_sel.clicked.connect(self._on_clear_selection)
        toolbar.addWidget(self.btn_clear_sel)

        root.addLayout(toolbar)

        # ---- 原图预览（左）+ 切割预览（右）----
        body = QHBoxLayout()
        body.setSpacing(12)

        # 原图
        self.src_view = ImageView("原图")
        self.src_view.setMinimumWidth(360)
        body.addWidget(self.src_view, 1)

        # 切割预览区（GraphicsView，可缩放平移）
        preview_wrap = QWidget()
        pv_layout = QVBoxLayout(preview_wrap)
        pv_layout.setContentsMargins(0, 0, 0, 0)
        pv_layout.setSpacing(6)

        header_row = QHBoxLayout()
        self.lbl_preview_title = QLabel("切割预览")
        self.lbl_preview_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        header_row.addWidget(self.lbl_preview_title)
        header_row.addStretch(1)
        self.lbl_zoom_hint = QLabel(
            "提示：双击=适应窗口 · Ctrl+滚轮缩放 · 拖动平移 · 单击选中 · Ctrl/Shift 多选"
        )
        self.lbl_zoom_hint.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_zoom_hint.setWordWrap(True)
        header_row.addWidget(self.lbl_zoom_hint, 1)
        self.lbl_selection_info = QLabel("已选: 0 张")
        self.lbl_selection_info.setStyleSheet(
            "color: #FFD23F; font-size: 12px; font-weight: bold; padding: 0 6px;"
        )
        header_row.addWidget(self.lbl_selection_info)
        pv_layout.addLayout(header_row)

        self.preview_view = PreviewView()
        pv_layout.addWidget(self.preview_view, 1)
        self.preview_view._scene.selectionChanged.connect(self._on_selection_changed)

        body.addWidget(preview_wrap, 1)
        root.addLayout(body, 1)

        # ---- 状态条 ----
        self.lbl_status = QLabel("尚未载入图片。")
        self.lbl_status.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self.lbl_status)

        self._tiles: list[np.ndarray] = []
        self._tile_items: list[QGraphicsPixmapItem] = []      # 每个 tile 的 pixmap item
        self._tile_borders: dict[int, QGraphicsRectItem] = {}  # tile 索引 -> 边框 item
        self._selected_indices: set[int] = set()
        self._refresh_preview_title()
        self._update_selection_label()

    # ------------------------------------------------------------------
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        try:
            from PIL import Image
            pil = Image.open(path).convert("RGBA")
            arr = np.array(pil)
        except Exception as exc:
            QMessageBox.critical(self, "打开失败", f"无法读取图片：\n{exc}")
            return
        self.load(arr, source_tab=f"切割: {path.split('/')[-1]}")

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
        self._source_arr = rgba
        self._source_tab = source_tab
        self.src_view.set_image(rgba)
        h, w = rgba.shape[:2]
        self.lbl_status.setText(
            f"原图: {w} × {h} px   |   来源: {source_tab or '(直接打开)'}   |   点「执行切割」生成预览"
        )
        self._tiles.clear()
        self._clear_preview()

    # ------------------------------------------------------------------
    def _refresh_preview_title(self) -> None:
        r = self.spin_rows.value()
        c = self.spin_cols.value()
        self.lbl_preview_title.setText(
            f"切割预览（待生成）：{r} × {c} = {r * c} 张"
        )

    def _recompute_tiles(self) -> None:
        """按当前行 × 列计算子图（不更新 UI）。"""
        self._tiles.clear()
        if self._source_arr is None:
            return
        rows = self.spin_rows.value()
        cols = self.spin_cols.value()
        h, w = self._source_arr.shape[:2]
        rh = h // rows
        rw = w // cols
        for ri in range(rows):
            for ci in range(cols):
                y0 = ri * rh
                y1 = h if ri == rows - 1 else (ri + 1) * rh
                x0 = ci * rw
                x1 = w if ci == cols - 1 else (ci + 1) * rw
                self._tiles.append(self._source_arr[y0:y1, x0:x1].copy())

    def _clear_preview(self) -> None:
        self.preview_view._scene.clear()
        self._tile_items.clear()
        self._tile_borders.clear()
        # _selected_indices 由 selectionChanged 信号自动清空
        self._update_selection_label()

    def _refresh_preview(self) -> None:
        """把 _tiles 按当前行 × 列渲染进 GraphicsScene。
        每个子图设为可选中，索引存在 UserRole，便于点击 / Ctrl-Shift 多选。
        """
        scene = self.preview_view._scene
        scene.clear()
        self._tile_items.clear()
        self._tile_borders.clear()

        if not self._tiles:
            self._update_selection_label()
            return

        cols = self.spin_cols.value()
        rows = self.spin_rows.value()
        tile_w = self._tiles[0].shape[1]
        tile_h = self._tiles[0].shape[0]
        gap = 8

        pen_normal = QPen(QColor(120, 200, 255), 1.5)
        pen_normal.setCosmetic(True)
        pen_selected = QPen(QColor(255, 210, 63), 3.0)
        pen_selected.setCosmetic(True)

        for i, tile in enumerate(self._tiles):
            r, c = divmod(i, cols)
            x = c * (tile_w + gap)
            y = r * (tile_h + gap)
            # 棋盘格背景（透明提示）
            pm = QPixmap(tile_w, tile_h)
            pm.fill(QColor(40, 40, 40))
            p = QPainter(pm)
            cell = 8
            for yy in range(0, tile_h, cell):
                for xx in range(0, tile_w, cell):
                    col = QColor(80, 80, 80) if ((yy // cell) + (xx // cell)) % 2 == 0 else QColor(50, 50, 50)
                    p.fillRect(xx, yy, cell, cell, col)
            p.end()
            tile_pix = numpy_to_qpixmap(tile)
            p2 = QPainter(pm)
            p2.drawPixmap(0, 0, tile_pix)
            p2.end()

            # 可选中的 pixmap item（索引存到 UserRole）
            item = QGraphicsPixmapItem(pm)
            item.setPos(x, y)
            item.setFlag(QGraphicsItem.ItemIsSelectable, True)
            item.setData(Qt.UserRole, i)
            scene.addItem(item)
            self._tile_items.append(item)

            # 边框（不可点选，跟随 tile 索引，便于切换高亮）
            rect_item = QGraphicsRectItem(QRectF(x, y, tile_w, tile_h))
            rect_item.setPen(pen_normal)
            rect_item.setBrush(QBrush(Qt.NoBrush))
            rect_item.setZValue(item.zValue() + 1)
            scene.addItem(rect_item)
            self._tile_borders[i] = rect_item

            # 标签
            label_item = QGraphicsTextItem(
                f"#{i + 1}  ({r + 1},{c + 1})  {tile.shape[1]}×{tile.shape[0]}"
            )
            label_item.setDefaultTextColor(QColor(220, 220, 220))
            label_item.setPos(x + 4, y + tile_h + 2)
            label_item.setZValue(item.zValue() + 2)
            scene.addItem(label_item)

        # 自适应场景矩形 + 默认初始视图
        total_w = cols * tile_w + (cols - 1) * gap
        total_h = rows * tile_h + (rows - 1) * gap + 20
        scene.setSceneRect(0, 0, total_w, total_h)
        self.preview_view.fit_to_view()
        self._update_selection_label()

    # ------------------------------------------------------------------
    def _on_cut(self) -> None:
        """点击「执行切割」：刷新一次预览，等同于参数变更。"""
        if self._source_arr is None:
            QMessageBox.information(self, "没有原图", "请先打开或载入一张图片。")
            return
        self._recompute_tiles()
        self._refresh_preview()
        self.lbl_status.setText(
            f"已生成 {len(self._tiles)} 张子图（{self.spin_rows.value()} × {self.spin_cols.value()}）"
        )

    # ------------------------------------------------------------------
    # 选择 / 高亮
    # ------------------------------------------------------------------
    def _update_selection_label(self) -> None:
        total = len(self._tile_items)
        sel = len(self._selected_indices)
        if total == 0:
            self.lbl_selection_info.setText("已选: 0 张")
        else:
            self.lbl_selection_info.setText(f"已选: {sel} / {total} 张")

    def _on_selection_changed(self) -> None:
        """scene.selectionChanged 回调：把当前选中映射成 tile 索引集合，更新高亮。"""
        new_sel: set[int] = set()
        for item in self.preview_view._scene.selectedItems():
            data = item.data(Qt.UserRole)
            if isinstance(data, int):
                new_sel.add(data)
        self._selected_indices = new_sel
        for idx, rect in self._tile_borders.items():
            if idx in self._selected_indices:
                pen = QPen(QColor(255, 210, 63), 3.0)
                pen.setCosmetic(True)
                rect.setPen(pen)
            else:
                pen = QPen(QColor(120, 200, 255), 1.5)
                pen.setCosmetic(True)
                rect.setPen(pen)
        self._update_selection_label()

    def _on_select_all(self) -> None:
        if not self._tile_items:
            return
        for item in self._tile_items:
            item.setSelected(True)

    def _on_clear_selection(self) -> None:
        self.preview_view._scene.clearSelection()

    def _on_fit_view(self) -> None:
        self.preview_view.fit_to_view()

    # ------------------------------------------------------------------
    # 把当前选中的子图加入暂存区
    # ------------------------------------------------------------------
    def _on_push_selected(self) -> None:
        if not self._tiles:
            QMessageBox.information(self, "没有子图", "请先切割一张图片。")
            return
        if not self._selected_indices:
            QMessageBox.information(self, "未选中", "请先在预览里点击要加入暂存区的子图。")
            return
        if self._buf is None:
            QMessageBox.warning(self, "暂存区不可用", "暂存区未初始化")
            return

        cols = self.spin_cols.value()
        rows = self.spin_rows.value()
        n = 0
        for i in sorted(self._selected_indices):
            r, c = divmod(i, cols)
            tag = f"切割 {rows}×{cols} 子图#{i + 1}(行{r + 1}列{c + 1})"
            self._buf.push(
                self._tiles[i],
                source_tab=self._source_tab or tag,
                source_file=tag,
            )
            n += 1
        self.lbl_status.setText(f"已将 {n} 张选中子图加入暂存区。")
        QMessageBox.information(self, "完成", f"已将 {n} 张子图加入右侧暂存区。")