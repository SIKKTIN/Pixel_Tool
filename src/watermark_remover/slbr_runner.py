"""SLBR 半透明水印自动检测与去除 —— 自 Test 项目移植并适配。

去掉了对 loguru / helper / image_output 的依赖,改用标准 logging + 内联函数。
模型结构引用通过 src/watermark_remover/slbr_runtime 路径注入实现。
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TILE_SIZE = 384
DEFAULT_TILE_BATCH = 4
MAX_TILE_BATCH = 32

# SLBR 运行时目录: 与 slbr_runner.py 同级下的 slbr_runtime/
RUNTIME_ROOT = Path(__file__).resolve().parent / "slbr_runtime"


@contextmanager
def _slbr_runtime_path():
    runtime_path = str(RUNTIME_ROOT)
    already_present = runtime_path in sys.path
    if not already_present:
        sys.path.insert(0, runtime_path)
    try:
        yield
    finally:
        if not already_present:
            try:
                sys.path.remove(runtime_path)
            except ValueError:
                pass


def _build_model_args(checkpoint_path: Path, device: torch.device):
    return SimpleNamespace(
        nets="slbr",
        models="slbr",
        name="slbr_v1",
        input_size=256,
        crop_size=256,
        checkpoint=str(checkpoint_path.parent),
        resume=str(checkpoint_path),
        evaluate=True,
        preprocess="resize",
        no_flip=True,
        mask_mode="res",
        bg_mode="res_mask",
        sim_metric="cos",
        k_center=2,
        project_mode="simple",
        use_refine=True,
        k_refine=3,
        k_skip_stage=3,
        gpu=device.type == "cuda",
        gpu_id="0",
    )


def clamp_tile_size(value) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TILE_SIZE
    return normalized if normalized in {256, 384, 512} else DEFAULT_TILE_SIZE


def clamp_tile_batch(value) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TILE_BATCH
    return max(1, min(MAX_TILE_BATCH, normalized))


def get_overlap_for_tile_size(tile_size: int) -> int:
    return max(1, int(tile_size) // 4)


def recommend_slbr_params(cuda_info: Optional[dict] = None) -> dict:
    cuda_info = cuda_info or {}
    if not cuda_info.get("cuda_available"):
        return {"tile_size": 256, "tile_batch": 1, "overlap": 64, "pad_multiple": 16}

    free = cuda_info.get("free_memory_mb")
    total = cuda_info.get("total_memory_mb")
    mem = float(free or total or 0)

    if mem >= 12000:
        ts, tb = 512, 4
    elif mem >= 8000:
        ts, tb = 384, 4
    elif mem >= 6000:
        ts, tb = 384, 3
    elif mem >= 4000:
        ts, tb = 384, 2
    elif mem >= 2000:
        ts, tb = 256, 4
    else:
        ts, tb = 256, 1

    return {"tile_size": ts, "tile_batch": tb, "overlap": get_overlap_for_tile_size(ts), "pad_multiple": 16}


def _image_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(image_rgb.transpose(2, 0, 1))


def _tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0, 1).numpy()
    image = (image.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _mask_to_bgr(mask: torch.Tensor) -> np.ndarray:
    mask = mask.detach().cpu().clamp(0, 1).numpy()
    mask = (mask.squeeze(0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _grid_canvas_size(length: int, tile_size: int, stride: int, pad_multiple: int) -> int:
    target = max(length, tile_size)
    if target > tile_size:
        steps = int(math.ceil((target - tile_size) / stride))
        target = tile_size + steps * stride
    if pad_multiple > 1:
        target = _ceil_to_multiple(target, pad_multiple)
        if target > tile_size:
            steps = int(math.ceil((target - tile_size) / stride))
            target = tile_size + steps * stride
    return target


def _pad_center_black(image: np.ndarray, tile_size: int, overlap: int, pad_multiple: int):
    stride = tile_size - overlap
    height, width = image.shape[:2]
    canvas_height = _grid_canvas_size(height, tile_size, stride, pad_multiple)
    canvas_width = _grid_canvas_size(width, tile_size, stride, pad_multiple)

    top = (canvas_height - height) // 2
    left = (canvas_width - width) // 2
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=image.dtype)
    canvas[top:top + height, left:left + width] = image
    return canvas, (top, left, height, width)


def _tile_positions(length: int, tile_size: int, stride: int):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _blend_weight(tile_size: int, overlap: int, min_weight: float = 0.05):
    if overlap <= 0:
        return torch.ones(1, tile_size, tile_size)

    weight = torch.ones(tile_size, dtype=torch.float32)
    ramp = torch.linspace(min_weight, 1.0, steps=overlap, dtype=torch.float32)
    weight[:overlap] = ramp
    weight[-overlap:] = ramp.flip(0)
    return weight.view(1, tile_size, 1) * weight.view(1, 1, tile_size)


def read_image_bgr(path: Path) -> np.ndarray:
    """读取图片文件为 BGR ndarray (支持中文路径)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


class SlbrRunner:
    """SLBR 自动去半透明水印模型."""

    def __init__(self, model_dir: str | Path, device: str | torch.device = "cpu"):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device = torch.device(device)
        self._model = None
        self._lock = threading.Lock()

    @property
    def checkpoint_path(self) -> Path:
        preferred = self.model_dir / "slbr.pth.tar"
        if preferred.is_file():
            return preferred
        return self.model_dir / "slbr" / "model_best.pth.tar"

    @property
    def installed(self) -> bool:
        return self.checkpoint_path.is_file()

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.installed:
                raise FileNotFoundError(
                    f"SLBR checkpoint not found: {self.checkpoint_path}"
                )

            logger.info("Loading SLBR checkpoint: %s", self.checkpoint_path)
            with _slbr_runtime_path():
                from src.networks.resunet import SLBR

                args = _build_model_args(self.checkpoint_path, self.device)
                model = SLBR(args=args, shared_depth=1, blocks=3, long_skip=True)
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
                state_dict = checkpoint.get("state_dict", checkpoint)
                model.load_state_dict(state_dict, strict=True)
                model.to(self.device)
                model.eval()
                self._model = model
            logger.info("SLBR model loaded")
            return self._model

    def _forward(self, batch: torch.Tensor):
        model = self._load_model()
        batch = batch.to(self.device).float()
        with torch.inference_mode():
            pred_images, pred_masks, _ = model(batch)
            pred_image = pred_images[0] if isinstance(pred_images, list) else pred_images
            pred_mask = pred_masks[0]
            final = pred_image * pred_mask + batch * (1 - pred_mask)
        return final.clamp(0, 1), pred_mask.clamp(0, 1)

    def infer_bgr(
        self,
        image_bgr: np.ndarray,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_batch: int = DEFAULT_TILE_BATCH,
        pad_multiple: int = 16,
    ):
        """SLBR 推理.

        Returns:
            (clean_bgr, mask_bgr): 处理后图片 + 检测到的水印蒙版
        """
        tile_size = clamp_tile_size(tile_size)
        tile_batch = clamp_tile_batch(tile_batch)
        overlap = get_overlap_for_tile_size(tile_size)
        stride = tile_size - overlap

        canvas_bgr, crop = _pad_center_black(image_bgr, tile_size, overlap, pad_multiple)
        canvas = _image_to_tensor(canvas_bgr)
        _, canvas_height, canvas_width = canvas.shape

        ys = _tile_positions(canvas_height, tile_size, stride)
        xs = _tile_positions(canvas_width, tile_size, stride)
        weight = _blend_weight(tile_size, overlap)

        clean_sum = torch.zeros(3, canvas_height, canvas_width, dtype=torch.float32)
        mask_sum = torch.zeros(1, canvas_height, canvas_width, dtype=torch.float32)
        weight_sum = torch.zeros(1, canvas_height, canvas_width, dtype=torch.float32)

        pending_tiles = []
        pending_coords = []

        def flush():
            if not pending_tiles:
                return
            batch = torch.stack(pending_tiles, dim=0)
            clean_batch, mask_batch = self._forward(batch)
            for clean_tile, mask_tile, (y, x) in zip(
                clean_batch.cpu(), mask_batch.cpu(), pending_coords
            ):
                clean_sum[:, y:y + tile_size, x:x + tile_size] += clean_tile * weight
                mask_sum[:, y:y + tile_size, x:x + tile_size] += mask_tile * weight
                weight_sum[:, y:y + tile_size, x:x + tile_size] += weight
            pending_tiles.clear()
            pending_coords.clear()

        for y in ys:
            for x in xs:
                pending_tiles.append(canvas[:, y:y + tile_size, x:x + tile_size])
                pending_coords.append((y, x))
                if len(pending_tiles) >= tile_batch:
                    flush()
        flush()

        clean = clean_sum / weight_sum.clamp_min(1e-6)
        mask = mask_sum / weight_sum.clamp_min(1e-6)
        top, left, height, width = crop
        clean = clean[:, top:top + height, left:left + width]
        mask = mask[:, top:top + height, left:left + width]
        return _tensor_to_bgr(clean), _mask_to_bgr(mask)
