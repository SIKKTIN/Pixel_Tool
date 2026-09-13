import unittest
import numpy as np

from perfect_pixel.black_white_matte import remove_background_black_white


class BlackWhiteMatteTests(unittest.TestCase):
    def test_reconstructs_transparent_and_opaque_pixels(self):
        fg = np.zeros((8, 10, 4), np.uint8)
        fg[..., :3] = [0, 0, 0]
        fg[..., 3] = 0
        fg[2:6, 2:8, :3] = [180, 80, 30]
        fg[2:6, 2:8, 3] = 255
        fg[3, 3, 3] = 128
        black = fg[..., :3].copy()
        white = fg[..., :3].copy()
        a = fg[..., 3:4].astype(np.float32) / 255
        black[:] = np.round(fg[..., :3] * a).astype(np.uint8)
        white[:] = np.round(fg[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
        result = remove_background_black_white(black, white)
        self.assertEqual(int(result[0, 0, 3]), 0)
        self.assertGreater(int(result[2, 2, 3]), 240)
        self.assertAlmostEqual(int(result[3, 3, 3]), 128, delta=3)
        self.assertLess(np.abs(result[3, 3, :3].astype(int) - [180, 80, 30]).max(), 8)

    def test_dimension_mismatch_fails(self):
        with self.assertRaises(ValueError):
            remove_background_black_white(np.zeros((2, 2, 3), np.uint8), np.zeros((3, 2, 3), np.uint8))
