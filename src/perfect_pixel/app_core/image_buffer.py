"""Shared image tray state used by the desktop tabs."""
from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal


class ImageBuffer(QObject):
    """Small process-local image store shared by all desktop tabs."""

    changed = Signal()

    def __init__(self, max_items: int = 20):
        super().__init__()
        self._items: list[dict] = []
        self._max_items = max_items
        self._active_id: Optional[str] = None

    def push(self, image: np.ndarray, source_tab: str = "", source_file: str = "") -> str:
        item_id = uuid.uuid4().hex[:8]
        h, w = image.shape[:2]
        self._items.append({
            "id": item_id, "image": image, "source_tab": source_tab,
            "source_file": source_file, "size_text": f"{w} × {h}",
            "created_at": len(self._items) + 1,
        })
        if len(self._items) > self._max_items:
            self._items.pop(0)
        self._active_id = item_id
        self.changed.emit()
        return item_id

    def set_active(self, item_id: str) -> None:
        if any(item["id"] == item_id for item in self._items):
            self._active_id = item_id
            self.changed.emit()

    def get_active(self) -> Optional[np.ndarray]:
        return next((item["image"] for item in self._items if item["id"] == self._active_id), None)

    def get_by_id(self, item_id: str) -> Optional[np.ndarray]:
        return next((item["image"] for item in self._items if item["id"] == item_id), None)

    def remove(self, item_id: str) -> None:
        self._items = [item for item in self._items if item["id"] != item_id]
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


_image_buffer: Optional[ImageBuffer] = None


def image_buffer() -> ImageBuffer:
    global _image_buffer
    if _image_buffer is None:
        _image_buffer = ImageBuffer()
    return _image_buffer
