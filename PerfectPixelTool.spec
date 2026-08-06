# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['desktop_app.py'],
    pathex=['src', 'src/watermark_remover/slbr_runtime'],
    binaries=[],
    datas=[('assets/app_icon.ico', 'assets'), ('assets/models', 'assets/models')],
    hiddenimports=['PIL._tkinter_finder', 'perfect_pixel.perfect_pixel', 'perfect_pixel.perfect_pixel_noCV2', 'src.networks.resunet', 'src.networks.blocks', 'src.networks.discriminator', 'src.networks.methods', 'src.models.SLBR', 'src.models.BasicModel', 'src.utils.model_init', 'src.utils.osutils', 'src.utils.imutils', 'src.utils.parallel', 'src.utils.losses', 'src.utils.misc', 'src.utils.transforms', 'pytorch_ssim', 'pytorch_iou', 'torch', 'torch.nn', 'torch.nn.functional', 'torch.utils', 'torch.utils.data', 'torchvision', 'torchvision.models', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'PySide6.QtNetwork', 'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets', 'PySide6.QtOpenGL', 'PySide6.QtPrintSupport', 'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtSvg', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'cv2', 'numpy', 'onnxruntime', 'PIL', 'PIL.Image'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PerfectPixelTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\app_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PerfectPixelTool',
)
