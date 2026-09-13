from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from perfect_pixel.app_core.image_io import load_rgba, normalize_rgba, save_png


class ImageIOTests(unittest.TestCase):
    def test_normalize_rgb_to_rgba(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        rgba = normalize_rgba(rgb)
        self.assertEqual(rgba.shape, (3, 4, 4))
        self.assertTrue(np.all(rgba[..., 3] == 255))

    def test_png_roundtrip_preserves_alpha_and_clears_hidden_rgb(self):
        image = np.zeros((4, 5, 4), dtype=np.uint8)
        image[..., :3] = 240
        image[1:3, 1:4, 3] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.png"
            save_png(path, image)
            result = load_rgba(path)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result[0, 0, 3], 0)
        self.assertEqual(int(result[0, 0, :3].sum()), 0)
        self.assertEqual(result[2, 2, 3], 255)
        self.assertEqual(result[2, 2, 0], 240)


if __name__ == "__main__":
    unittest.main()
