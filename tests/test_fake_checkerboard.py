import numpy as np

from perfect_pixel.fake_checkerboard import remove_fake_checkerboard


def checker(h=32, w=40, tile=4):
    out = np.zeros((h, w, 4), np.uint8)
    yy, xx = np.indices((h, w))
    parity = ((yy // tile + xx // tile) & 1)
    out[..., :3] = np.where(parity[..., None] == 0, 238, 205)
    out[..., 3] = 255
    return out


def test_checkerboard_removed_and_foreground_preserved():
    img = checker()
    img[10:23, 12:28, :3] = [220, 40, 30]
    result = remove_fake_checkerboard(img, tile_size=4, tolerance=12, binary_alpha=True)
    assert result.shape == img.shape and result.dtype == np.uint8
    assert np.all(result[:8, :8, 3] == 0)
    assert np.all(result[14:20, 16:24, 3] == 255)
    assert np.all(result[result[..., 3] == 0, :3] == 0)


def test_existing_alpha_and_soft_edge():
    img = checker(tile=5)
    img[..., 3] = 255
    img[12:20, 15:25, :3] = [100, 100, 100]
    img[12:20, 15:25, 3] = 180
    result = remove_fake_checkerboard(img, tile_size=5, tolerance=10, anti_alias=True)
    assert np.any((result[..., 3] > 0) & (result[..., 3] < 255))
    assert np.all(result[0, 0, 3] == 0)


def test_auto_tile_detection():
    img = checker(tile=4)
    img[11:21, 14:24, :3] = [20, 180, 70]
    result = remove_fake_checkerboard(img, tolerance=14, binary_alpha=True)
    assert result[15, 18, 3] == 255
    assert result[0, 0, 3] == 0
