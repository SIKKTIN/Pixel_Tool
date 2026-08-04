@echo off
REM ============================================================
REM   Perfect Pixel Tool  --  开发调试运行 (不打包)
REM
REM   用法: 双击 dev.bat,或在项目根目录执行 dev.bat
REM   特点: 直接用 python 跑,改代码后 Ctrl+C 停止,再双击即可重跑。
REM         不涉及 PyInstaller,速度秒开。
REM
REM   打包: 回到 build.bat (约 9 分钟)
REM ============================================================

cd /d "%~dp0"

REM 模型目录: 默认指向 Test 项目共享模型
REM 如果 PerfectPixelTool\models\ 下已有模型,会自动优先使用
set PERFECTPIXEL_MODEL_DIR=%~dp0models
if not exist "%PERFECTPIXEL_MODEL_DIR%\big-lama.pt" (
    if exist "..\Test\src\models\big-lama.pt" (
        set PERFECTPIXEL_MODEL_DIR=..\Test\src\models
    )
)

REM 入口程序: python   = 带控制台(可见 print/报错,适合调试)
REM           pythonw  = 无控制台(干净无黑窗,适合发布预览)
set ENTRY=python

echo.
echo Perfect Pixel Tool -- 开发模式
echo   模型目录: %PERFECTPIXEL_MODEL_DIR%
echo   入口程序: %ENTRY% (改为 pythonw 可关闭控制台)
echo.
%ENTRY% desktop_app.py
echo.
echo [退出] 按任意键关闭...
pause >nul
