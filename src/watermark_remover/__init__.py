"""WatermarkRemover — 从 Test 项目移植的纯 Python 算法层。

提供 SLBR / LaMa 两个模型的本地推理接口，不依赖 FastAPI / Electron / Pydantic 等 Web 栈。
"""

from .lama_model import LaMaModel
from .slbr_runner import SlbrRunner

__all__ = ["LaMaModel", "SlbrRunner"]
