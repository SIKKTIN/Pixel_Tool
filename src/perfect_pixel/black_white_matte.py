"""Recover RGBA foreground from matched black/white composites."""
from __future__ import annotations

import numpy as np

from .app_core.image_io import normalize_rgba


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    v = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _to_srgb(rgb: np.ndarray) -> np.ndarray:
    v = np.clip(rgb, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def remove_background_black_white(
    black: np.ndarray,
    white: np.ndarray,
    *,
    background_black: tuple[int, int, int] = (0, 0, 0),
    background_white: tuple[int, int, int] = (255, 255, 255),
    anti_alias: bool = True,
    binary_alpha: bool = False,
    edge_shrink: int = 0,
) -> np.ndarray:
    """Reconstruct a foreground from two aligned composites."""
    b = normalize_rgba(black)[..., :3]
    w = normalize_rgba(white)[..., :3]
    if b.shape != w.shape:
        raise ValueError("black and white images must have identical dimensions")
    # PNG compositors commonly blend encoded channel values directly. Use the
    # observed space here so generated black/white reference images round-trip
    # exactly; linear helpers remain available for future calibrated inputs.
    bl = b.astype(np.float32) / 255.0
    wl = w.astype(np.float32) / 255.0
    bb = np.asarray(background_black, dtype=np.float32).reshape(1, 1, 3) / 255.0
    bw = np.asarray(background_white, dtype=np.float32).reshape(1, 1, 3) / 255.0
    denom = bw - bb
    valid = np.abs(denom) > 1e-6
    alpha_channels = 1.0 - np.divide(wl - bl, denom, out=np.zeros_like(wl), where=valid)
    alpha = np.median(alpha_channels, axis=2)
    alpha = np.clip(alpha, 0.0, 1.0)
    # Recover foreground from whichever composite is numerically stable.
    f_channels = np.divide(bl - (1.0 - alpha[..., None]) * bb, alpha[..., None],
                           out=np.zeros_like(bl), where=alpha[..., None] > 1e-5)
    unstable = alpha < 1e-5
    f_channels[unstable] = 0.0
    out_alpha = np.round(alpha * 255.0).astype(np.uint8)
    if binary_alpha:
        out_alpha = np.where(out_alpha >= 128, 255, 0).astype(np.uint8)
    elif not anti_alias:
        out_alpha = np.where(out_alpha >= 128, 255, 0).astype(np.uint8)
    out_rgb = np.round(np.clip(f_channels, 0.0, 1.0) * 255.0).astype(np.uint8)
    out = np.dstack((out_rgb, out_alpha))
    for _ in range(max(0, int(edge_shrink))):
        solid = out_alpha > 0
        edge = np.zeros_like(solid)
        edge[1:] |= ~solid[:-1]; edge[:-1] |= ~solid[1:]
        edge[:, 1:] |= ~solid[:, :-1]; edge[:, :-1] |= ~solid[:, 1:]
        out_alpha[solid & edge] = np.minimum(out_alpha[solid & edge], 96 if anti_alias and not binary_alpha else 0)
    if binary_alpha or not anti_alias:
        out_alpha = np.where(out_alpha >= 128, 255, 0).astype(np.uint8)
    out[..., 3] = out_alpha
    out[out_alpha == 0, :3] = 0
    return np.ascontiguousarray(out)


def reconstruction_error(
    black: np.ndarray, white: np.ndarray, result: np.ndarray,
    *, background_black=(0, 0, 0), background_white=(255, 255, 255),
) -> np.ndarray:
    """Return per-pixel RGB reconstruction error in 0..255."""
    b = normalize_rgba(black)[..., :3].astype(np.float32)
    w = normalize_rgba(white)[..., :3].astype(np.float32)
    r = normalize_rgba(result)
    a = r[..., 3:4].astype(np.float32) / 255.0
    bb = np.asarray(background_black, dtype=np.float32).reshape(1, 1, 3)
    bw = np.asarray(background_white, dtype=np.float32).reshape(1, 1, 3)
    pred_b = r[..., :3].astype(np.float32) * a + bb * (1.0 - a)
    pred_w = r[..., :3].astype(np.float32) * a + bw * (1.0 - a)
    return (np.abs(pred_b - b).mean(axis=2) + np.abs(pred_w - w).mean(axis=2)) * 0.5


__all__ = ["remove_background_black_white", "reconstruction_error"]
