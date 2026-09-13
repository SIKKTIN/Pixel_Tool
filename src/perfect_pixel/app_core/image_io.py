"""Consistent image loading and PNG/JPEG export helpers."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def load_rgba(path: str | Path) -> np.ndarray:
    """Load any supported image as contiguous uint8 RGBA."""
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGBA"), dtype=np.uint8))


def load_rgb(path: str | Path) -> np.ndarray:
    """Load any supported image as contiguous uint8 RGB."""
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def normalize_rgba(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 3:
        arr = np.dstack([arr, np.full(arr.shape[:2], 255, dtype=np.uint8)])
    if arr.shape[2] != 4:
        raise ValueError("image must have 1, 3, or 4 channels")
    return np.ascontiguousarray(arr)


def save_png(path: str | Path, image: np.ndarray) -> Path:
    arr = normalize_rgba(image)
    # Remove hidden RGB values from fully transparent pixels.
    arr = arr.copy()
    arr[arr[..., 3] == 0, :3] = 0
    Image.fromarray(arr, mode="RGBA").save(path, "PNG")
    return Path(path)
