"""图标流水线主脚本。

用法:
    python regen_ico.py                        # 生成 app_icon.ico
    python regen_ico.py --verify              # 生成并验证各层
    python regen_ico.py --check-exe PATH       # 从 EXE 抽图标并验证

生成逻辑  : PNG-in-ICO（Pillow 内置，多尺寸: 16/24/32/48/64/128/256）
旧版 BMP  : 见 gen_ico.py（手写二进制，DIB header，适合极端兼容性场景）
"""
import argparse
import os
import sys

from PIL import Image


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "assets", "app_icon_src.png")
DST  = os.path.join(ROOT, "assets", "app_icon.ico")
SIZES = [16, 24, 32, 48, 64, 128, 256]


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

def generate(src: str = SRC, dst: str = DST, sizes: list[int] = SIZES) -> None:
    img = Image.open(src).convert("RGBA")
    img.save(dst, format="ICO", sizes=[(s, s) for s in sizes])
    print(f"OK: {os.path.getsize(dst) // 1024} KB  |  sizes: {sizes}")


# ---------------------------------------------------------------------------
# 验证（内嵌 verify_ico.py 逻辑）
# ---------------------------------------------------------------------------

def verify(dst: str = DST) -> None:
    """读取 ICO，每层存为 PNG 并打印尺寸。"""
    ico = Image.open(dst)
    print(f"ICO  mode={ico.mode}  size={ico.size}  format={ico.format}  frames={ico.n_frames}")
    out_dir = os.path.join(ROOT, "dist")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(ico.n_frames):
        ico.seek(i)
        w, h = ico.size
        out = os.path.join(out_dir, f"ico_layer_{w}.png")
        ico.convert("RGBA").save(out)
        print(f"  layer {i+1}: {w}x{h}  ->  {out}")


# ---------------------------------------------------------------------------
# 从 EXE 抽图标（内嵌 check_icon.ps1 逻辑）
# ---------------------------------------------------------------------------

def check_exe(exe_path: str) -> None:
    """从 EXE 抽取图标并另存为 PNG。"""
    import subprocess, tempfile

    ps1 = r"""
Add-Type -AssemblyName System.Drawing
$exe = $Args[0]
$out  = $Args[1]
try {
    $ic = [System.Drawing.Icon]::ExtractAssociatedIcon($exe)
    Write-Host ("Icon size: " + $ic.Size)
    $bmp = $ic.ToBitmap()
    $bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Host ("Saved: " + $bmp.Width + "x" + $bmp.Height)
} catch {
    Write-Host ("Error: " + $_.Exception.Message)
    exit 1
}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ps1", delete=False,
                                     encoding="utf-8") as f:
        f.write(ps1)
        ps1_path = f.name

    out_png = os.path.join(ROOT, "dist", "exe_icon.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    try:
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps1_path,
             exe_path, out_png],
            capture_output=True, text=True
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
    finally:
        os.unlink(ps1_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="生成 ICO 后验证各层并导出为 PNG")
    parser.add_argument("--check-exe", metavar="EXE",
                        help="从 EXE 抽取图标并验证")
    args = parser.parse_args()

    if args.check_exe:
        check_exe(args.check_exe)
    else:
        generate()
        if args.verify:
            verify()


if __name__ == "__main__":
    main()
