"""Remove baked-in checkerboard transparency backgrounds from RGBA images.

The routine is deliberately dependency-free (NumPy only).  It estimates the
checker period from the image border, classifies pixels against the two border
colours, and removes only the connected background component.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .app_core.image_io import normalize_rgba


def _estimate_period(rgb: np.ndarray) -> int:
    h, w = rgb.shape[:2]
    # Average narrow border bands; the foreground is usually sparse there.
    bands = []
    k = min(8, h, w)
    bands.extend([rgb[:k].mean(axis=(0, 1)), rgb[-k:].mean(axis=(0, 1))])
    hs = np.median(rgb[:k], axis=0)
    vs = np.median(rgb[:, :k], axis=1) if w >= k else rgb.mean(axis=1)
    # Use luminance signatures, which remain stable for subtly tinted grids.
    sigs = [hs @ np.array([.299, .587, .114]), vs @ np.array([.299, .587, .114])]
    scores: list[tuple[float, int]] = []
    for p in range(2, min(128, max(h, w) // 2) + 1):
        vals = []
        for sig in sigs:
            if len(sig) > p:
                vals.append(float(np.mean(np.abs(sig[p:] - sig[:-p]))))
        if vals:
            scores.append((sum(vals) / len(vals), p))
    if not scores:
        return 8
    # A checkerboard repeats after two tiles.  Choose the first strong local
    # minimum to avoid selecting a large multiple of the true period.
    scores.sort()
    p = scores[0][1]
    return max(1, int(round(p / 2)))


def _border_samples(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    n = max(2, min(12, min(h, w) // 8 or 2))
    mask = np.zeros((h, w), bool)
    mask[:n] = mask[-n:] = True
    mask[:, :n] = mask[:, -n:] = True
    mask &= alpha > 0
    samples = rgb[mask]
    if len(samples) == 0:
        samples = rgb.reshape(-1, 3)
    return samples.astype(np.float32)


def _two_colors(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Tiny two-means implementation avoids a scikit-learn dependency.
    c0 = samples[np.argmin(samples.sum(axis=1))].copy()
    c1 = samples[np.argmax(samples.sum(axis=1))].copy()
    for _ in range(8):
        d0 = np.sum((samples - c0) ** 2, axis=1)
        d1 = np.sum((samples - c1) ** 2, axis=1)
        a = d0 <= d1
        if a.any(): c0 = samples[a].mean(axis=0)
        if (~a).any(): c1 = samples[~a].mean(axis=0)
    return c0, c1


def _connected_from_edge(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if mask[0, x]: q.append((0, x)); seen[0, x] = True
        if h > 1 and mask[h - 1, x]: q.append((h - 1, x)); seen[h - 1, x] = True
    for y in range(1, h - 1):
        if mask[y, 0]: q.append((y, 0)); seen[y, 0] = True
        if w > 1 and mask[y, w - 1]: q.append((y, w - 1)); seen[y, w - 1] = True
    while q:
        y, x = q.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True; q.append((yy, xx))
    return seen


def remove_fake_checkerboard(
    image: np.ndarray,
    *,
    tile_size: int | None = None,
    tolerance: float = 18.0,
    anti_alias: bool = True,
    binary_alpha: bool = False,
) -> np.ndarray:
    """Remove a baked checkerboard background and return contiguous RGBA uint8."""
    if binary_alpha:
        anti_alias = False
    arr = normalize_rgba(image).copy()
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3].copy()
    h, w = alpha.shape
    t = max(1, int(tile_size or _estimate_period(arr[..., :3])))
    c0, c1 = _two_colors(_border_samples(arr[..., :3], alpha))
    d0 = np.sqrt(np.sum((rgb - c0) ** 2, axis=2))
    d1 = np.sqrt(np.sum((rgb - c1) ** 2, axis=2))
    close = np.minimum(d0, d1) <= float(tolerance) * np.sqrt(3.0)
    # Checkerboard phase is used as a weak prior; colour match remains primary.
    # This also handles crops that begin in either checker phase.
    phase = ((np.indices((h, w))[0] // t + np.indices((h, w))[1] // t) & 1)
    cdist = np.where(phase == 0, d0, d1)
    checker = close | (cdist <= float(tolerance) * np.sqrt(3.0) * 1.35)
    bg = _connected_from_edge(checker & (alpha > 0))
    out_alpha = alpha.copy()
    out_alpha[bg] = 0
    if anti_alias:
        # Soften only colour-matched pixels immediately adjacent to removed bg.
        nbr = np.zeros_like(bg)
        nbr[1:] |= bg[:-1]; nbr[:-1] |= bg[1:]
        nbr[:, 1:] |= bg[:, :-1]; nbr[:, :-1] |= bg[:, 1:]
        soft = nbr & ~bg & close
        out_alpha[soft] = np.minimum(out_alpha[soft], 128)
    if binary_alpha:
        out_alpha = np.where(out_alpha >= 128, 255, 0).astype(np.uint8)
    arr[..., 3] = out_alpha
    arr[out_alpha == 0, :3] = 0
    return np.ascontiguousarray(arr)


__all__ = ["remove_fake_checkerboard"]
