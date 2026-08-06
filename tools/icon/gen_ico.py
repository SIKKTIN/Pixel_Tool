"""旧版 BMP-in-ICO 生成器（手写二进制格式）。

新项目请用 regen_ico.py（Pillow PNG-in-ICO）。
本脚本保留用于：
  - 极端兼容性场景（某些老旧程序只认 BMP-in-ICO）
  - 理解 ICO 文件格式（BINARY_ENCODING_NOTES.md）

源图: assets/app_icon_src.png
输出: assets/app_icon.ico
尺寸 : 16 / 32 / 48 / 64 / 128 / 256
"""
from PIL import Image
import struct
import os

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "app_icon_src.png")
DST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "app_icon.ico")
SIZES = [16, 32, 48, 64, 128, 256]


class DIBHeader:
    """BITMAPINFOHEADER (40 bytes) — ICO/DIB 格式头。"""

    def __init__(self, w: int, h: int, bpp: int = 32):
        self.size   = 40
        self.width  = w
        self.height = h * 2   # height × 2: 包含 AND mask 行
        self.planes = 1
        self.bpp    = bpp
        self.compression = 0
        self.raw_size    = 0
        self.h_res       = 0
        self.v_res       = 0
        self.colors      = 0
        self.important   = 0

    def to_bytes(self) -> bytes:
        return struct.pack(
            "<IiiHHIIiiII",
            self.size, self.width, self.height, self.planes,
            self.bpp, self.compression, self.raw_size,
            self.h_res, self.v_res, self.colors, self.important,
        )


def make_bmp_layer(img: Image.Image, size: int) -> bytes:
    """把 RGBA 转成 BMP-encoded DIB（含 AND mask），返回裸字节。"""
    resized = img.resize((size, size), Image.LANCZOS)
    px = resized.load()

    pixel_rows = bytearray()
    mask_rows  = bytearray()

    # 像素行：BGRA bottom-up，4 字节对齐
    for y in range(size - 1, -1, -1):
        row = bytearray()
        for x in range(size):
            r, g, b, a = px[x, y]
            row.extend([b, g, r, a])          # BGRA
        row.extend(b"\x00" * ((4 - len(row) % 4) % 4))   # 4-byte pad
        pixel_rows.extend(row)

    # AND mask: 0 = 透明由 alpha 控制
    for y in range(size - 1, -1, -1):
        mrow = bytearray()
        for x in range(0, size, 8):
            mrow.append(0)
        mrow.extend(b"\x00" * ((4 - len(mrow) % 4) % 4))
        mask_rows.extend(mrow)

    return DIBHeader(size, size).to_bytes() + bytes(pixel_rows) + bytes(mask_rows)


def build(src: str, dst: str, sizes: list[int]) -> None:
    img = Image.open(src).convert("RGBA")

    # ICONDIR header
    header = struct.pack("<HHH", 0, 1, len(sizes))

    # 每条 ICONDIRENTRY 16 bytes
    entries: list[bytes]  = []
    images:  list[bytes]  = []
    offset   = 6 + len(sizes) * 16

    for s in sizes:
        bmp  = make_bmp_layer(img, s)
        w    = 0 if s >= 256 else s
        h    = 0 if s >= 256 else s
        entries.append(struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(bmp), offset))
        images.append(bmp)
        offset += len(bmp)

    with open(dst, "wb") as f:
        f.write(header)
        for e in entries:
            f.write(e)
        for b in images:
            f.write(b)

    print(f"OK: {os.path.getsize(dst) // 1024} KB  |  {len(sizes)} sizes  |  BMP-encoded")


if __name__ == "__main__":
    build(SRC, DST, SIZES)
