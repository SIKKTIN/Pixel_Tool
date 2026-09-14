"""序列帧预览 Tab：把一张 Sprite Sheet 按网格切分并实时播放。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QFormLayout, QGroupBox,
)


class SequencePreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sheet: np.ndarray | None = None
        self._frames: list[np.ndarray] = []
        self._index = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        root = QVBoxLayout(self)
        self._shared_controls = QWidget()
        shared_layout = QVBoxLayout(self._shared_controls)
        shared_layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QVBoxLayout()
        self.open_btn = QPushButton("打开序列图")
        self.open_btn.clicked.connect(self.open_image)
        toolbar.addWidget(self.open_btn)
        toolbar.addWidget(QLabel("列数"))
        self.cols = QSpinBox(); self.cols.setRange(1, 128); self.cols.setValue(8)
        self.cols.valueChanged.connect(self.rebuild_frames); toolbar.addWidget(self.cols)
        toolbar.addWidget(QLabel("行数"))
        self.rows = QSpinBox(); self.rows.setRange(1, 128); self.rows.setValue(1)
        self.rows.valueChanged.connect(self.rebuild_frames); toolbar.addWidget(self.rows)
        toolbar.addWidget(QLabel("帧率"))
        self.fps = QDoubleSpinBox(); self.fps.setRange(1, 60); self.fps.setValue(8); self.fps.setSuffix(" FPS")
        self.fps.valueChanged.connect(self.update_timer); toolbar.addWidget(self.fps)
        self.loop = QCheckBox("循环播放"); self.loop.setChecked(True); toolbar.addWidget(self.loop)
        toolbar.addStretch(1); shared_layout.addLayout(toolbar)

        self.preview = QLabel("请打开一张横向或网格序列图")
        self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumSize(0, 0)
        self.preview.setStyleSheet("background:#202020; color:#aaa; border:1px solid #555;")
        root.addWidget(self.preview, 1)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("▶ 播放"); self.play_btn.setEnabled(False); self.play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_btn)
        self.frame_slider = QSlider(Qt.Horizontal); self.frame_slider.setEnabled(False); self.frame_slider.valueChanged.connect(self.seek)
        controls.addWidget(self.frame_slider, 1)
        self.info = QLabel("未加载")
        controls.addWidget(self.info); shared_layout.addLayout(controls)
        root.addWidget(self._shared_controls)

    def detach_shared_controls(self):
        self._shared_controls.setParent(None)
        return self._shared_controls

    def on_open(self):
        self.open_image()

    def on_run(self):
        self.rebuild_frames()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开序列图", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            try:
                im = Image.open(path).convert("RGBA")
                self._sheet = np.array(im)
                self.rebuild_frames()
            except Exception as exc:
                self.info.setText(f"打开失败: {exc}")

    def rebuild_frames(self, *_):
        if self._sheet is None: return
        h, w = self._sheet.shape[:2]; cols, rows = self.cols.value(), self.rows.value()
        fw, fh = w // cols, h // rows
        if fw < 1 or fh < 1: return
        self._frames = [self._sheet[r*fh:(r+1)*fh, c*fw:(c+1)*fw] for r in range(rows) for c in range(cols)]
        self._index = 0; self.frame_slider.setRange(0, len(self._frames)-1); self.frame_slider.setEnabled(True)
        self.play_btn.setEnabled(True); self.update_timer(); self.show_frame()

    def update_timer(self, *_):
        self._timer.setInterval(max(1, round(1000 / self.fps.value())))

    def toggle_play(self):
        if self._timer.isActive(): self._timer.stop(); self.play_btn.setText("▶ 播放")
        else: self._timer.start(); self.play_btn.setText("⏸ 暂停")

    def _next_frame(self):
        if not self._frames: return
        if self._index + 1 >= len(self._frames):
            if not self.loop.isChecked(): self._timer.stop(); self.play_btn.setText("▶ 播放"); return
            self._index = 0
        else: self._index += 1
        self.frame_slider.blockSignals(True); self.frame_slider.setValue(self._index); self.frame_slider.blockSignals(False); self.show_frame()

    def seek(self, value):
        self._index = value; self.show_frame()

    def show_frame(self):
        if not self._frames: return
        arr = np.ascontiguousarray(self._frames[self._index]); h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        self.info.setText(f"第 {self._index + 1}/{len(self._frames)} 帧  {w}×{h}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._frames: self.show_frame()
