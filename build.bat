@echo off
REM ============================================================
REM   Perfect Pixel Tool  —  一键打包成单文件 .exe
REM
REM   用法: 双击 build.bat,或在项目根目录执行 build.bat
REM   输出: dist\PerfectPixelTool.exe
REM ============================================================

setlocal

echo.
echo [1/4] 检查 Python 环境...
python --version || goto :err_python

echo.
echo [2/4] 安装/更新依赖...
python -m pip install --upgrade pip >nul
python -m pip install -e . >nul
python -m pip install pyinstaller PySide6 >nul
if errorlevel 1 goto :err_pip

echo.
echo [3/4] 清理旧的构建产物...
if exist build rmdir /S /Q build
if exist dist  rmdir /S /Q dist
if exist PerfectPixelTool.spec del /Q PerfectPixelTool.spec

echo.
echo [4/4] 调用 PyInstaller 打包(单文件、无控制台、窗口模式)...
echo       这一步会持续 1~3 分钟,请耐心等待...
python -m PyInstaller ^
    --noconsole ^
    --onefile ^
    --windowed ^
    --name PerfectPixelTool ^
    --collect-submodules PySide6 ^
    --collect-submodules perfect_pixel ^
    desktop_app.py

if errorlevel 1 goto :err_pyinstaller

echo.
echo ============================================================
echo   打包成功!
echo   EXE 位置: dist\PerfectPixelTool.exe
echo   直接双击即可运行。
echo ============================================================

REM 可选:自动启动测试
set /p launch="是否现在启动测试? [y/N] "
if /i "%launch%"=="y" start "" dist\PerfectPixelTool.exe

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