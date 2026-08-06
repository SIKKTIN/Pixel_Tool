@echo off
REM ============================================================
REM   Perfect Pixel Tool  --  一键打包 (onedir 模式)
REM
REM   用法: 双击 build.bat,或在项目根目录执行 build.bat
REM   输出: dist\PerfectPixelTool\PerfectPixelTool.exe (含 _internal/)
REM   模型: 打包完成后会自动从 Test 项目复制到 dist\PerfectPixelTool\models\
REM
REM   说明: --onedir 模式 (vs --onefile) 产生标准应用目录结构,
REM         避免单文件 EXE 写入时与 Windows 索引器的锁冲突。
REM         运行前请进入 dist\PerfectPixelTool\ 双击 EXE。
REM ============================================================

setlocal

cd /d "%~dp0"

echo.
echo [1/5] 检查 Python 环境...
python --version || goto :err_python

echo.
echo [2/5] 检查打包依赖 (PyInstaller / PySide6)...
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo       安装 PyInstaller...
    python -m pip install pyinstaller --quiet || goto :err_pip
)
python -c "import PySide6" 2>nul
if errorlevel 1 (
    echo       安装 PySide6...
    python -m pip install PySide6 --quiet || goto :err_pip
)

echo.
echo [3/5] 清理旧的构建产物...
if exist build rmdir /S /Q build
if exist dist  rmdir /S /Q dist
if exist PerfectPixelTool.spec del /Q PerfectPixelTool.spec

echo.
echo [4/5] 调用 PyInstaller 打包 (onedir、无控制台、窗口模式)...
echo       这一步会持续 3~8 分钟 (含 torch),请耐心等待...
python -m PyInstaller ^
    --noconsole ^
    --onedir ^
    --windowed ^
    --name PerfectPixelTool ^
    --icon=assets\app_icon.ico ^
    --paths=src ^
    --paths=src/watermark_remover/slbr_runtime ^
    --add-data=assets/app_icon.ico;assets ^
    --add-data=assets/models;assets/models ^
    --hidden-import=PIL._tkinter_finder ^
    --hidden-import=perfect_pixel.perfect_pixel ^
    --hidden-import=perfect_pixel.perfect_pixel_noCV2 ^
    --hidden-import=src.networks.resunet ^
    --hidden-import=src.networks.blocks ^
    --hidden-import=src.networks.discriminator ^
    --hidden-import=src.networks.methods ^
    --hidden-import=src.models.SLBR ^
    --hidden-import=src.models.BasicModel ^
    --hidden-import=src.utils.model_init ^
    --hidden-import=src.utils.osutils ^
    --hidden-import=src.utils.imutils ^
    --hidden-import=src.utils.parallel ^
    --hidden-import=src.utils.losses ^
    --hidden-import=src.utils.misc ^
    --hidden-import=src.utils.transforms ^
    --hidden-import=pytorch_ssim ^
    --hidden-import=pytorch_iou ^
    --hidden-import=torch ^
    --hidden-import=torch.nn ^
    --hidden-import=torch.nn.functional ^
    --hidden-import=torch.utils ^
    --hidden-import=torch.utils.data ^
    --hidden-import=torchvision ^
    --hidden-import=torchvision.models ^
    --hidden-import=PySide6.QtCore ^
    --hidden-import=PySide6.QtGui ^
    --hidden-import=PySide6.QtWidgets ^
    --hidden-import=PySide6.QtNetwork ^
    --hidden-import=PySide6.QtMultimedia ^
    --hidden-import=PySide6.QtMultimediaWidgets ^
    --hidden-import=PySide6.QtOpenGL ^
    --hidden-import=PySide6.QtPrintSupport ^
    --hidden-import=PySide6.QtQml ^
    --hidden-import=PySide6.QtQuick ^
    --hidden-import=PySide6.QtSvg ^
    --hidden-import=PySide6.QtWebEngineCore ^
    --hidden-import=PySide6.QtWebEngineWidgets ^
    --hidden-import=PySide6.QtWidgets ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=onnxruntime ^
    --hidden-import=PIL ^
    --hidden-import=PIL.Image ^
    desktop_app.py

if errorlevel 1 goto :err_pyinstaller

echo.
echo [5/5] 复制模型文件到 EXE 同目录 (sidecar 模式)...
if not exist dist\PerfectPixelTool\models mkdir dist\PerfectPixelTool\models
if exist "..\Test\src\models\big-lama.pt" (
    copy /Y "..\Test\src\models\big-lama.pt" dist\PerfectPixelTool\models\ >nul && echo       [OK] big-lama.pt
) else (
    echo       [WARN] 未找到 ..\Test\src\models\big-lama.pt
)
if exist "..\Test\src\models\slbr.pth.tar" (
    copy /Y "..\Test\src\models\slbr.pth.tar" dist\PerfectPixelTool\models\ >nul && echo       [OK] slbr.pth.tar
) else (
    echo       [WARN] 未找到 ..\Test\src\models\slbr.pth.tar
)

echo.
echo ============================================================
echo   打包成功!
echo   EXE 位置:    dist\PerfectPixelTool\PerfectPixelTool.exe
echo   运行时目录:  dist\PerfectPixelTool\        (含 _internal\ 子目录)
echo   模型位置:    dist\PerfectPixelTool\models\
echo.
echo   注意: 必须保留整个 dist\PerfectPixelTool\ 目录一起分发,
echo         进入该目录双击 PerfectPixelTool.exe 运行。
echo ============================================================

REM 可选:自动启动测试
set /p launch="是否现在启动测试? [y/N] "
if /i "%launch%"=="y" start "" dist\PerfectPixelTool\PerfectPixelTool.exe

endlocal
exit /b 0

:err_python
echo [错误] 未检测到 Python,请先安装 Python 3.9+ 并加入 PATH。
pause
exit /b 1

:err_pip
echo [错误] 依赖安装失败,请检查网络/源配置。
pause
exit /b 1

:err_pyinstaller
echo [错误] PyInstaller 打包失败,请查看上方日志。
pause
exit /b 1
