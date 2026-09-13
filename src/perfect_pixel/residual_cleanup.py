"""Second-pass cleanup for opaque speckles and matte fringes."""
from __future__ import annotations

import numpy as np

from .app_core.image_io import normalize_rgba


def _components(mask: np.ndarray, minimum: int) -> tuple[np.ndarray, int]:
    cleaned = mask.copy()
    seen = np.zeros_like(mask, bool)
    removed = 0
    h, w = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        pixels = []
        while stack:
            y, x = stack.pop(); pixels.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; stack.append((yy, xx))
        if len(pixels) < minimum:
            removed += len(pixels)
            for y, x in pixels: cleaned[y, x] = False
    return cleaned, removed


def cleanup_background_residuals(
    image: np.ndarray,
    *,
    min_component_size: int = 4,
    edge_strength: float = 28.0,
    edge_radius: int = 1,
    anti_alias: bool = True,
) -> np.ndarray:
    """Remove small islands and bright matte pixels along an existing cutout."""
    arr = normalize_rgba(image).copy()
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3].copy()
    foreground, _ = _components(alpha > 0, max(1, int(min_component_size)))
    alpha[~foreground] = 0
    radius = max(0, int(edge_radius))
    for _ in range(radius):
        solid = alpha > 0
        near_empty = np.zeros_like(solid)
        near_empty[1:] |= ~solid[:-1]; near_empty[:-1] |= ~solid[1:]
        near_empty[:, 1:] |= ~solid[:, :-1]; near_empty[:, :-1] |= ~solid[:, 1:]
        # A matte fringe is usually a bright/low-saturation edge pixel whose
        # neighbouring foreground is materially darker. Preserve coloured art.
        lum = rgb @ np.array([.299, .587, .114], dtype=np.float32)
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        inner = np.zeros_like(lum)
        count = np.zeros_like(lum)
        bright_neighbors = np.zeros_like(lum)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                shifted = np.roll(np.roll(lum, dy, axis=0), dx, axis=1)
                valid = np.roll(np.roll(solid, dy, axis=0), dx, axis=1)
                inner += np.where(valid, shifted, 0); count += valid
                bright_neighbors += valid & (shifted > 150)
        inner = inner / np.maximum(count, 1)
        # Small baked squares are often connected to the sprite by one pixel,
        # so component filtering alone cannot remove them. Detect a neutral
        # bright pixel with a dark local neighbourhood as a matte speck.
        # Larger UI values mean stronger cleanup. Keep a small minimum contrast
        # so 255 does not become a no-op or erase every bright highlight.
        contrast_threshold = max(4.0, 36.0 - float(edge_strength) * 0.125)
        speck = (lum > 175) & (chroma < 90) & (bright_neighbors <= 3) & ((lum - inner) > contrast_threshold)
        fringe = solid & (near_empty | speck) & speck
        alpha[fringe] = np.minimum(alpha[fringe], 96 if anti_alias else 0)
    if not anti_alias:
        alpha = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    arr[..., 3] = alpha
    arr[alpha == 0, :3] = 0
    return np.ascontiguousarray(arr)


__all__ = ["cleanup_background_residuals"]
