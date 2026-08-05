"""
背景去除核心算法集 — 对标 image-master (image.moonrailgun.com)

提供三种模式:
  - Color:   基于颜色的洪水填充 + Chroma Key 净化
  - Channel: 基于颜色通道（Luminance / Saturation / R / G / B）的阈值抠图
  - AI:      iSnNet 语义分割（可选，需本地 ONNX 模型）

全部在 CPU + NumPy 下实现，无外部网络请求。
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Literal, Tuple


# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

RGB = Tuple[int, int, int]
ChannelSource = Literal["red", "green", "blue", "luminance", "saturation"]
AIModel = Literal["isnet", "isnet_fp16", "isnet_quint8"]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = _clamp01((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _color_distance(c1: np.ndarray, c2: np.ndarray | RGB) -> float:
    """欧氏 RGB 距离（标量或向量）。"""
    dr = c1[..., 0] - np.asarray(c2[0])
    dg = c1[..., 1] - np.asarray(c2[1])
    db = c1[..., 2] - np.asarray(c2[2])
    return float(np.sqrt(dr * dr + dg * dg + db * db).max())


def _get_channel_value(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                       channel: ChannelSource) -> np.ndarray:
    """提取指定通道值，支持向量输入。"""
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    if channel == "red":
        return r
    if channel == "green":
        return g
    if channel == "blue":
        return b
    if channel == "luminance":
        return 0.299 * r + 0.587 * g + 0.114 * b
    # saturation
    vmax = np.maximum(np.maximum(r, g), b)
    vmin = np.minimum(np.minimum(r, g), b)
    delta = vmax - vmin
    denom = np.where(vmax == 0, 1.0, vmax)
    return (delta / denom) * 255.0


def _detect_chroma_key_channel(bg_color: RGB) -> Optional[Literal["red", "green", "blue"]]:
    """检测背景主通道（R/G/B 之一），用于 Chroma Key 净化。"""
    r, g, b = bg_color
    channels = sorted([("red", r), ("green", g), ("blue", b)], key=lambda x: x[1], reverse=True)
    dominant = channels[0]
    runner_up = channels[1]
    dominance = dominant[1] - runner_up[1]
    if dominant[1] < 64 or dominance < 24:
        return None
    return dominant[0]  # type: ignore[return-value]


def _chroma_key_alpha(
    pixel_rgb: np.ndarray,
    bg_color: RGB,
    tolerance_distance: float,
    original_alpha: np.ndarray,
) -> np.ndarray:
    """计算 Chroma Key 细化后的 alpha。pixel_rgb / original_alpha 形状相同（N,3）/（N,）。"""
    channel = _detect_chroma_key_channel(bg_color)
    if channel is None or (original_alpha <= 0).all():
        return original_alpha

    r_p, g_p, b_p = pixel_rgb[:, 0], pixel_rgb[:, 1], pixel_rgb[:, 2]
    r_b, g_b, b_b = float(bg_color[0]), float(bg_color[1]), float(bg_color[2])

    if channel == "red":
        bg_dom = r_b - max(g_b, b_b)
        pix_dom = np.maximum(0.0, r_p - np.maximum(g_p, b_p))
    elif channel == "green":
        bg_dom = g_b - max(r_b, b_b)
        pix_dom = np.maximum(0.0, g_p - np.maximum(r_p, b_p))
    else:
        bg_dom = b_b - max(r_b, g_b)
        pix_dom = np.maximum(0.0, b_p - np.maximum(r_p, g_p))

    if bg_dom < 24:
        return original_alpha

    distance_limit = max(tolerance_distance * 1.35, 48.0)

    dist = np.sqrt(
        (r_p - r_b) ** 2 + (g_p - g_b) ** 2 + (b_p - b_b) ** 2
    )
    color_match = np.clip(1.0 - dist / distance_limit, 0.0, 1.0)
    dominance_match = np.clip(pix_dom / bg_dom, 0.0, 1.0)
    removal_signal = color_match * dominance_match

    # alpha_factor = 1 - smoothstep(0.12, 0.88, signal)
    alpha_factor = 1.0 - _smoothstep(0.12, 0.88, removal_signal)
    new_alpha = np.round(np.clip(original_alpha * alpha_factor, 0, 255))
    return np.where(removal_signal > 0.12, new_alpha, original_alpha)


def _suppress_chroma_spill(
    data: np.ndarray,
    pixel_idx: np.ndarray,
    channel: Literal["red", "green", "blue"],
    strength: float,
) -> None:
    """减少主通道溢色（如绿幕边缘的绿色残留）。"""
    safe_strength = _clamp01(strength)
    if safe_strength <= 0:
        return
    ci = 0 if channel == "red" else (1 if channel == "green" else 2)
    other = [0, 1, 2]
    other.remove(ci)
    cv = data.flat[pixel_idx * 4 + ci].astype(np.float32)
    mo = np.maximum(data.flat[pixel_idx * 4 + other[0]].astype(np.float32),
                     data.flat[pixel_idx * 4 + other[1]].astype(np.float32))
    spill = np.maximum(0.0, cv - mo)
    data.flat[pixel_idx * 4 + ci] = np.clip(cv - spill * safe_strength * 0.6, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 距离变换（8连通 Chamfer）
# ---------------------------------------------------------------------------

def _distance_to_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """两遍 Chamfer 距离变换。mask 为 bool/uint8，True/1 表示透明区域。"""
    dist = np.full((height, width), np.inf, dtype=np.float32)
    dist[mask > 0] = 0.0

    # Forward pass
    for y in range(height):
        for x in range(width):
            if dist[y, x] == 0:
                continue
            best = dist[y, x]
            if x > 0:
                best = min(best, dist[y, x - 1] + 1)
            if y > 0:
                best = min(best, dist[y - 1, x] + 1)
            if x > 0 and y > 0:
                best = min(best, dist[y - 1, x - 1] + 1.414)
            if x < width - 1 and y > 0:
                best = min(best, dist[y - 1, x + 1] + 1.414)
            dist[y, x] = best

    # Backward pass
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if dist[y, x] == 0:
                continue
            best = dist[y, x]
            if x < width - 1:
                best = min(best, dist[y, x + 1] + 1)
            if y < height - 1:
                best = min(best, dist[y + 1, x] + 1)
            if x < width - 1 and y < height - 1:
                best = min(best, dist[y + 1, x + 1] + 1.414)
            if x > 0 and y < height - 1:
                best = min(best, dist[y + 1, x - 1] + 1.414)
            dist[y, x] = best

    return dist


# ---------------------------------------------------------------------------
# 洪水填充（栈式 4连通）
# ---------------------------------------------------------------------------

def _flood_fill_remove(
    data: np.ndarray,
    visited: np.ndarray,
    height: int,
    width: int,
    start_x: int,
    start_y: int,
    bg_color: RGB,
    tolerance_distance: float,
) -> None:
    """从种子点向外洪水填充，标记 visited 并将 alpha 设为 0。原地修改 data。"""
    if visited[start_y, start_x]:
        return
    stack: list[Tuple[int, int]] = [(start_x, start_y)]

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if visited[y, x]:
            continue

        dr = float(data[y, x, 0]) - bg_color[0]
        dg = float(data[y, x, 1]) - bg_color[1]
        db = float(data[y, x, 2]) - bg_color[2]
        if np.sqrt(dr * dr + dg * dg + db * db) > tolerance_distance:
            continue

        visited[y, x] = 1
        data[y, x, 3] = 0

        stack.append((x - 1, y))
        stack.append((x + 1, y))
        stack.append((x, y - 1))
        stack.append((x, y + 1))


# ---------------------------------------------------------------------------
# 边缘收缩（Edge Shrink）
# ---------------------------------------------------------------------------

def _apply_edge_shrink(
    data: np.ndarray,
    height: int,
    width: int,
    radius: float,
) -> None:
    """利用距离变换向内侵蚀前景边缘。"""
    if radius <= 0:
        return
    visited = (data[:, :, 3] == 0).astype(np.uint8)
    dist = _distance_to_mask(visited, height, width)

    soft_zone = min(1.0, radius * 0.5)
    mask_solid = dist <= radius - soft_zone
    mask_soft = (dist > radius - soft_zone) & (dist < radius)

    data[mask_solid, 3] = 0

    if mask_soft.any():
        t = (dist[mask_soft] - (radius - soft_zone)) / soft_zone
        smooth = t * t * (3.0 - 2.0 * t)
        data[mask_soft, 3] = np.round(data[mask_soft, 3] * smooth).astype(np.uint8)


# ---------------------------------------------------------------------------
# 边缘羽化（Feather）
# ---------------------------------------------------------------------------

def _apply_feather(
    data: np.ndarray,
    height: int,
    width: int,
    radius: float,
) -> None:
    """基于距离变换的 alpha 余弦衰减羽化。"""
    if radius <= 0:
        return
    alpha = data[:, :, 3].astype(np.float32)
    dist = _distance_to_mask(alpha == 0, height, width)

    mask = (alpha > 0) & (dist < radius)
    t = dist[mask] / radius
    falloff = (1.0 - np.cos(t * np.pi)) / 2.0
    alpha[mask] = np.round(alpha[mask] * falloff)
    data[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Alpha 高斯模糊
# ---------------------------------------------------------------------------

def _gaussian_blur_alpha(
    data: np.ndarray,
    height: int,
    width: int,
    sigma: float,
) -> None:
    """仅对 alpha 通道做可分离高斯模糊。"""
    if sigma <= 0:
        return
    radius = int(np.ceil(sigma * 2.5))
    kernel_size = radius * 2 + 1
    x = np.arange(kernel_size, dtype=np.float32) - radius
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()

    alpha = data[:, :, 3].astype(np.float32)

    # 水平
    temp = np.zeros_like(alpha)
    for y in range(height):
        for x_idx in range(width):
            val = 0.0
            for k in range(kernel_size):
                sx = max(0, min(width - 1, x_idx + k - radius))
                val += alpha[y, sx] * kernel[k]
            temp[y, x_idx] = val

    # 垂直
    for y in range(height):
        for x_idx in range(width):
            val = 0.0
            for k in range(kernel_size):
                sy = max(0, min(height - 1, y + k - radius))
                val += temp[sy, x_idx] * kernel[k]
            data[y, x_idx, 3] = int(np.clip(val, 0, 255))


# ---------------------------------------------------------------------------
# 边界柔化
# ---------------------------------------------------------------------------

def _apply_boundary_softening(
    data: np.ndarray,
    visited: np.ndarray,
    height: int,
    width: int,
    bg_color: RGB,
    tolerance_distance: float,
) -> None:
    """在洪水填充边缘做空间+颜色双重平滑过渡。"""
    band = 2.5
    outer_tolerance = tolerance_distance * 1.5
    dist = _distance_to_mask(visited, height, width)

    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            if dist[y, x] > band:
                continue

            dr = float(data[y, x, 0]) - bg_color[0]
            dg = float(data[y, x, 1]) - bg_color[1]
            db = float(data[y, x, 2]) - bg_color[2]
            color_dist = np.sqrt(dr * dr + dg * dg + db * db)

            if color_dist >= outer_tolerance:
                continue

            spatial_t = dist[y, x] / band
            color_t = color_dist / outer_tolerance
            t = max(spatial_t, color_t)
            smooth = t * t * (3.0 - 2.0 * t)
            data[y, x, 3] = int(round(float(data[y, x, 3]) * smooth))


# ---------------------------------------------------------------------------
# Chroma Key 细化
# ---------------------------------------------------------------------------

def _apply_chroma_key_refinement(
    data: np.ndarray,
    height: int,
    width: int,
    bg_color: RGB,
    tolerance_distance: float,
    removed_mask: np.ndarray,
) -> None:
    """边缘区域执行 Chroma Key alpha 降低 + 溢色抑制。"""
    channel = _detect_chroma_key_channel(bg_color)
    if channel is None:
        return

    edge_band = 2.5
    dist = _distance_to_mask(removed_mask, height, width)

    for y in range(height):
        for x in range(width):
            if dist[y, x] > edge_band:
                continue
            original_alpha = data[y, x, 3]
            if original_alpha == 0:
                continue

            pixel_rgb = data[y, x, :3]
            next_alpha = _chroma_key_alpha(
                pixel_rgb[np.newaxis, :],
                bg_color,
                tolerance_distance,
                np.array([original_alpha], dtype=np.float32),
            )[0]

            if next_alpha >= original_alpha:
                continue

            _suppress_chroma_spill(data, np.array([y * width + x]), channel,
                                   1.0 - next_alpha / original_alpha)
            data[y, x, 3] = int(next_alpha)


# ---------------------------------------------------------------------------
# 模式一：按颜色去背景
# ---------------------------------------------------------------------------

def remove_background_color(
    image: np.ndarray,
    *,
    tolerance: float = 10.0,
    contiguous_only: bool = True,
    target_color: Optional[RGB] = None,
    feather: float = 0.0,
    anti_alias: bool = True,
    chroma_key: bool = True,
    seed_points: Optional[list[Tuple[int, int]]] = None,
    edge_shrink: float = 0.0,
) -> np.ndarray:
    """
    基于颜色的背景去除。

    Args:
        image:       HxWx3 或 HxWx4 uint8 图片（会自动转 RGBA）
        tolerance:   颜色容差 [1, 100]，百分比映射到 RGB 欧氏距离阈值
        contiguous_only: True = 仅从四边洪水填充；False = 全图匹配
        target_color: 手动指定背景色，默认取左上角
        feather:     边缘羽化半径（像素）
        anti_alias:  是否启用边缘抗锯齿
        chroma_key:  是否启用 Chroma Key 净化（适合绿幕/蓝幕）
        seed_points: 封闭区域洪水填充种子点列表 [(x, y), ...]
        edge_shrink: 前景边缘向内收缩半径
    Returns:
        HxWx4 uint8 RGBA 图片
    """
    # 统一为 RGBA
    if image.shape[2] == 3:
        data = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])
    else:
        data = image.copy()
    data = np.ascontiguousarray(data, dtype=np.uint8)

    height, width = data.shape[:2]
    r_ch = data[:, :, 0].astype(np.float32)
    g_ch = data[:, :, 1].astype(np.float32)
    b_ch = data[:, :, 2].astype(np.float32)

    # 背景色
    if target_color is None:
        bg_color: RGB = (int(r_ch[0, 0]), int(g_ch[0, 0]), int(b_ch[0, 0]))
    else:
        bg_color = target_color

    max_dist = 441.67
    tolerance_distance = (tolerance / 100.0) * max_dist

    visited = np.zeros((height, width), dtype=np.uint8)

    # ---- 洪水填充 ----
    if contiguous_only:
        for x in range(width):
            _flood_fill_remove(data, visited, height, width, x, 0, bg_color, tolerance_distance)
            _flood_fill_remove(data, visited, height, width, x, height - 1, bg_color, tolerance_distance)
        for y in range(height):
            _flood_fill_remove(data, visited, height, width, 0, y, bg_color, tolerance_distance)
            _flood_fill_remove(data, visited, height, width, width - 1, y, bg_color, tolerance_distance)
    else:
        if anti_alias:
            inner_t = tolerance_distance * 0.85
            outer_t = tolerance_distance * 1.15
            for y in range(height):
                for x in range(width):
                    idx = y * width + x
                    dr = float(data[y, x, 0]) - bg_color[0]
                    dg = float(data[y, x, 1]) - bg_color[1]
                    db = float(data[y, x, 2]) - bg_color[2]
                    dist = np.sqrt(dr * dr + dg * dg + db * db)
                    if dist <= inner_t:
                        data[y, x, 3] = 0
                        visited[y, x] = 1
                    elif dist < outer_t:
                        t = (dist - inner_t) / (outer_t - inner_t)
                        smooth = t * t * (3.0 - 2.0 * t)
                        data[y, x, 3] = int(round(float(data[y, x, 3]) * smooth))
        else:
            for y in range(height):
                for x in range(width):
                    dr = float(data[y, x, 0]) - bg_color[0]
                    dg = float(data[y, x, 1]) - bg_color[1]
                    db = float(data[y, x, 2]) - bg_color[2]
                    if np.sqrt(dr * dr + dg * dg + db * db) <= tolerance_distance:
                        data[y, x, 3] = 0
                        visited[y, x] = 1

    # ---- 种子点 ----
    if seed_points:
        for (sx, sy) in seed_points:
            if 0 <= sx < width and 0 <= sy < height:
                seed_rgb: RGB = (int(r_ch[sy, sx]), int(g_ch[sy, sx]), int(b_ch[sy, sx]))
                _flood_fill_remove(data, visited, height, width, sx, sy, seed_rgb, tolerance_distance)

    # ---- 边界柔化 ----
    if anti_alias:
        _apply_boundary_softening(data, visited, height, width, bg_color, tolerance_distance)
        _gaussian_blur_alpha(data, height, width, sigma=0.8)

    # ---- Chroma Key 细化 ----
    if chroma_key:
        _apply_chroma_key_refinement(data, height, width, bg_color, tolerance_distance, removed_mask=visited)

    # ---- 后处理 ----
    if edge_shrink > 0:
        _apply_edge_shrink(data, height, width, edge_shrink)
    if feather > 0:
        _apply_feather(data, height, width, feather)

    return data


# ---------------------------------------------------------------------------
# 模式二：按通道抠图
# ---------------------------------------------------------------------------

def remove_background_channel(
    image: np.ndarray,
    *,
    channel: ChannelSource = "luminance",
    min_threshold: float = 0.0,
    max_threshold: float = 255.0,
    invert: bool = False,
    feather: float = 0.0,
    edge_shrink: float = 0.0,
) -> np.ndarray:
    """
    基于颜色通道值的阈值抠图。

    Args:
        image:         HxWx3 或 HxWx4 uint8
        channel:       通道来源 ("red"|"green"|"blue"|"luminance"|"saturation")
        min_threshold: 通道值下界
        max_threshold: 通道值上界
        invert:        True = 区间外保留；False = 区间内保留
        feather:       边缘羽化
        edge_shrink:   前景收缩
    Returns:
        HxWx4 uint8 RGBA
    """
    if image.shape[2] == 3:
        data = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])
    else:
        data = image.copy()
    data = np.ascontiguousarray(data, dtype=np.uint8)

    height, width = data.shape[:2]
    r_ch = data[:, :, 0].astype(np.float32)
    g_ch = data[:, :, 1].astype(np.float32)
    b_ch = data[:, :, 2].astype(np.float32)

    ch_vals = _get_channel_value(r_ch, g_ch, b_ch, channel)

    # 软边宽度
    soft_edge = max(min((max_threshold - min_threshold) * 0.08, 8.0), 1.0)

    lut = np.zeros(256, dtype=np.float32)
    for v in range(256):
        if v < min_threshold - soft_edge or v > max_threshold + soft_edge:
            alpha = 0.0
        elif v < min_threshold:
            t = (v - (min_threshold - soft_edge)) / soft_edge
            alpha = (t * t * (3.0 - 2.0 * t)) * 255.0
        elif v > max_threshold:
            t = ((max_threshold + soft_edge) - v) / soft_edge
            alpha = (t * t * (3.0 - 2.0 * t)) * 255.0
        else:
            alpha = 255.0

        lut[v] = alpha if not invert else (255.0 - alpha)

    orig_alpha = data[:, :, 3].astype(np.float32)
    new_alpha = np.round((orig_alpha / 255.0) * lut[np.clip(ch_vals, 0, 255).astype(np.uint8)])
    data[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)

    if edge_shrink > 0:
        _apply_edge_shrink(data, height, width, edge_shrink)
    if feather > 0:
        _apply_feather(data, height, width, feather)

    return data


# ---------------------------------------------------------------------------
# 模式三：AI（留空，需用户提供本地 ONNX 模型路径）
# ---------------------------------------------------------------------------

def remove_background_ai(
    image: np.ndarray,
    *,
    model_path: str,
    edge_shrink: float = 0.0,
) -> np.ndarray:
    """
    调用本地 ONNX iSnNet 模型做语义分割抠图。

    Args:
        image:      HxWx3 或 HxWx4 uint8
        model_path: ONNX 模型文件路径（需提前下载 isnet.onnx）
        edge_shrink: 前景收缩
    Returns:
        HxWx4 uint8 RGBA
    """
    # 此处依赖 onnxruntime；框架保留，用户自行下载模型
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "onnxruntime 未安装。请 pip install onnxruntime ，"
            "并下载 isnet.onnx 模型文件。"
        )

    # 预处理：HWC -> CHW，归一化
    if image.shape[2] == 4:
        rgb = image[:, :, :3]
    else:
        rgb = image

    rgb = cv2.resize(rgb, (1024, 1024))
    rgb = rgb.astype(np.float32) / 255.0
    rgb = rgb.transpose(2, 0, 1)[np.newaxis, ...]  # (1,3,1024,1024)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    mask: np.ndarray = sess.run([out_name], {inp_name: rgb})[0][0, 0]  # (1024,1024)
    mask = (mask * 255).clip(0, 255).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    alpha = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    if image.shape[2] == 4:
        alpha[:, :, :3] = image[:, :, :3]
    else:
        alpha[:, :, :3] = image
    alpha[:, :, 3] = mask

    height, width = alpha.shape[:2]
    if edge_shrink > 0:
        _apply_edge_shrink(alpha, height, width, edge_shrink)

    return alpha


# ---------------------------------------------------------------------------
# 快捷入口
# ---------------------------------------------------------------------------

def remove_background(
    image: np.ndarray,
    mode: Literal["color", "channel", "ai"] = "color",
    **kwargs,
) -> np.ndarray:
    """
    统一入口，自动分发到对应模式。

    典型用法：
        result = remove_background(img, mode="color", tolerance=10, chroma_key=True)
        result = remove_background(img, mode="channel", channel="luminance",
                                 min_threshold=10, max_threshold=200)
    """
    if mode == "channel":
        return remove_background_channel(image, **kwargs)
    elif mode == "ai":
        return remove_background_ai(image, **kwargs)
    else:
        return remove_background_color(image, **kwargs)
