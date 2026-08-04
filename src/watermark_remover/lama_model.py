"""LaMa 推理模型 —— 自 Test 项目移植并适配。

去掉了对 loguru 的依赖，使用标准 logging。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_env_model_dir = os.environ.get("PERFECTPIXEL_MODEL_DIR", "").strip()


def _exe_dir() -> Path:
    """返回 EXE 所在目录(PyInstaller --onefile 模式下)。

    普通 Python 运行时返回 __file__ 的父目录。
    """
    try:
        # PyInstaller 解压后写入 sys.executable = EXE 路径
        exe = Path(sys.executable).resolve()
        if getattr(sys, "_MEIPASS", None):
            return exe.parent
    except Exception:
        pass
    return Path(__file__).resolve().parent


# 模型路径解析优先级:
#   1. 环境变量 PERFECTPIXEL_MODEL_DIR
#   2. EXE 同目录的 models/  (PyInstaller 打包后)
#   3. 项目内 models/  (PerfectPixelTool/models, 源码运行时)
#   4. 上层 ../models/  (Test 项目共享, 兼容旧布局)
_DEFAULT_MODEL_DIR_CANDIDATES = [
    Path(p) for p in (
        _env_model_dir,
        str(_exe_dir() / "models"),
        str(Path(__file__).resolve().parents[2] / "models"),
        str(Path(__file__).resolve().parents[3] / "Test" / "src" / "models"),
    ) if p
]


def _resolve_model_dir() -> Path:
    # 优先选择同时存在 LaMa 和 SLBR 权重的目录
    has_both = lambda d: (d / "big-lama.pt").is_file() and (d / "slbr.pth.tar").is_file()
    for candidate in _DEFAULT_MODEL_DIR_CANDIDATES:
        if candidate and has_both(candidate):
            return candidate
    # 次选:任一模型存在
    for candidate in _DEFAULT_MODEL_DIR_CANDIDATES:
        if candidate and candidate.is_dir() and (
            (candidate / "big-lama.pt").is_file() or (candidate / "slbr.pth.tar").is_file()
        ):
            return candidate
    # 兜底:返回首选,让上层报清晰的"模型未找到"错误
    return _DEFAULT_MODEL_DIR_CANDIDATES[0] if _DEFAULT_MODEL_DIR_CANDIDATES else Path("models")


MODEL_DIR = _resolve_model_dir()
LAMA_MODEL_PATH = MODEL_DIR / "big-lama.pt"
LAMA_MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"


def norm_img(np_img):
    """Normalize image for LaMa model."""
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img: np.ndarray, mod: int, square: bool = False, min_size: int = None):
    """Pad image to be divisible by mod."""
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def load_jit_model(model_path, device):
    """Load JIT traced model."""
    logger.info("Loading LaMa model from %s", model_path)
    model = torch.jit.load(model_path, map_location="cpu").to(device)
    model.eval()
    return model


class LaMaModel:
    """LaMa 蒙版修复模型 (big-lama.pt)."""

    name = "lama"
    pad_mod = 8

    def __init__(self, model_path: str | Path | None = None, device: torch.device | None = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model_path = Path(model_path) if model_path else LAMA_MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"LaMa model not found at {self.model_path}")
        self.model = load_jit_model(self.model_path, self.device)
        logger.info("LaMa model loaded on %s", self.device)

    def __call__(self, image_rgb: np.ndarray, mask_gray: np.ndarray):
        """Run LaMa inpainting.

        Args:
            image_rgb: H x W x 3 RGB uint8
            mask_gray: H x W grayscale uint8, 255=待修复区域

        Returns:
            result_bgr: H x W x 3 BGR uint8
        """
        origin_height, origin_width = image_rgb.shape[:2]

        pad_image = pad_img_to_modulo(image_rgb, mod=self.pad_mod)
        pad_mask = pad_img_to_modulo(mask_gray, mod=self.pad_mod)

        image = norm_img(pad_image)
        mask = norm_img(pad_mask)
        mask = (mask > 0) * 1

        image_t = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            inpainted = self.model(image_t, mask_t)

        result = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype("uint8")
        result = result[:origin_height, :origin_width]
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    @classmethod
    def is_installed(cls, model_path: str | Path | None = None) -> bool:
        path = Path(model_path) if model_path else LAMA_MODEL_PATH
        return path.is_file()
