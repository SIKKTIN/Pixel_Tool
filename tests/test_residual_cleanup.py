import numpy as np
import unittest

from perfect_pixel.residual_cleanup import cleanup_background_residuals


def test_residual_cleanup_removes_island_and_clears_hidden_rgb():
    img = np.zeros((24, 24, 4), np.uint8)
    img[..., :3] = [120, 120, 120]
    img[..., 3] = 0
    img[5:19, 7:17, :3] = [120, 40, 20]
    img[5:19, 7:17, 3] = 255
    img[2, 2, :3] = [255, 255, 255]
    img[2, 2, 3] = 255
    result = cleanup_background_residuals(img, min_component_size=4, edge_radius=0)
    assert result[2, 2, 3] == 0
    assert result[10, 10, 3] == 255
    assert np.all(result[result[..., 3] == 0, :3] == 0)


def test_residual_cleanup_binary_alpha():
    img = np.zeros((12, 12, 4), np.uint8)
    img[3:9, 3:9, :3] = [220, 220, 220]
    img[3:9, 3:9, 3] = 180
    result = cleanup_background_residuals(img, min_component_size=1, anti_alias=False)
    assert set(np.unique(result[..., 3])).issubset({0, 255})


class ResidualCleanupTests(unittest.TestCase):
    def test_island_and_hidden_rgb(self):
        test_residual_cleanup_removes_island_and_clears_hidden_rgb()

    def test_binary_alpha(self):
        test_residual_cleanup_binary_alpha()
