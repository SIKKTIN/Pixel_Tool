"""Check the watermark tab and local models; --inference also runs both models.

Run with the same Python interpreter used by dev.bat:
    .venv/Scripts/python.exe tools/check_watermark.py --inference
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", action="store_true", help="Run SLBR and LaMa on a small RGBA image")
    args = parser.parse_args()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    import numpy as np
    import torch
    from PIL import Image
    from PySide6.QtWidgets import QApplication
    from desktop_app import MainWindow
    from perfect_pixel.app_core.image_buffer import image_buffer
    from watermark_remover.lama_model import MODEL_DIR
    from watermark_remover.widget import LamaWorker, SlbrWorker, WatermarkWidget

    print(f"Python: {sys.executable}", flush=True)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"PyTorch: {torch.__version__}; device: {device}", flush=True)
    print(f"Models: {MODEL_DIR}", flush=True)
    for name in ("big-lama.pt", "slbr.pth.tar"):
        if not (MODEL_DIR / name).is_file():
            raise FileNotFoundError(MODEL_DIR / name)

    app = QApplication([])
    window = MainWindow()
    widget = window.watermark_tab
    if not isinstance(widget, WatermarkWidget):
        raise RuntimeError("The watermark tab loaded an error placeholder")
    print("PASS watermark tab construction and model paths", flush=True)
    if args.inference:
        source = np.zeros((65, 79, 4), dtype=np.uint8)
        source[..., :3] = (110, 160, 90)
        source[4:-4, 4:-4, 3] = 255
        source[4, 4:-4, 3] = 128
        source[28:36, 30:45, :3] = 240
        mask = np.zeros(source.shape[:2], dtype=np.uint8)
        mask[26:38, 28:47] = 255
        widget.load_from_buffer(source)
        workers = (
            ("SLBR", SlbrWorker(source, tile_size=256, tile_batch=1), widget._on_slbr_done),
            ("LaMa", LamaWorker(source, mask), widget._on_lama_done),
        )
        for name, worker, callback in workers:
            errors = []
            completed = []
            worker.failed.connect(errors.append)
            worker.finished_ok.connect(callback)
            worker.finished_ok.connect(lambda *result: completed.append(True))
            print(f"Running {name}...", flush=True)
            worker.run()
            app.processEvents()
            if errors or not completed:
                raise RuntimeError(f"{name} failed: {errors}")
            result = widget.result_image
            assert result.shape == source.shape and result.dtype == np.uint8
            np.testing.assert_array_equal(result[..., 3], source[..., 3])
            np.testing.assert_array_equal(image_buffer().get_active(), result)
            with tempfile.TemporaryDirectory(prefix="watermark-check-") as folder:
                path = Path(folder) / "result.png"
                Image.fromarray(result).save(path)
                with Image.open(path) as saved:
                    np.testing.assert_array_equal(np.array(saved), result)
            print(f"PASS {name}: inference, RGBA, tray and PNG roundtrip", flush=True)
    window.close()


if __name__ == "__main__":
    main()
