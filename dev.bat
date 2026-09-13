@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "ENTRY="
set "FALLBACK_ENTRY="
set "RUN_RESULT=0"

REM Prefer the project environment and existing system installations.
if defined PERFECTPIXEL_PYTHON call :try_python "%PERFECTPIXEL_PYTHON%"
if not defined ENTRY call :try_python "%~dp0.venv\Scripts\python.exe"
if not defined ENTRY call :try_python "%~dp0venv\Scripts\python.exe"
if not defined ENTRY for /f "delims=" %%P in ('where python 2^>nul') do call :try_python "%%P"
if not defined ENTRY for /f "tokens=2,*" %%A in ('reg query HKCU\Software\Python\PythonCore /s /v ExecutablePath 2^>nul ^| findstr /c:"REG_SZ"') do call :try_python "%%B"
if not defined ENTRY for /f "tokens=2,*" %%A in ('reg query HKLM\Software\Python\PythonCore /s /v ExecutablePath 2^>nul ^| findstr /c:"REG_SZ"') do call :try_python "%%B"
if not defined ENTRY call :try_python "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not defined ENTRY if defined FALLBACK_ENTRY goto :no_complete_environment
if not defined ENTRY goto :missing_python

set "PERFECTPIXEL_MODEL_DIR=%~dp0models"
if not exist "%PERFECTPIXEL_MODEL_DIR%\big-lama.pt" if exist "%~dp0..\Test\src\models\big-lama.pt" set "PERFECTPIXEL_MODEL_DIR=%~dp0..\Test\src\models"

echo.
echo Perfect Pixel Tool -- Development
echo Python: "%ENTRY%"
echo Models: "%PERFECTPIXEL_MODEL_DIR%"
echo.
"%ENTRY%" -c "import cv2, PySide6, numpy, PIL"
if errorlevel 1 goto :missing_dependencies
if /i "%~1"=="--check" goto :finish
"%ENTRY%" "%~dp0desktop_app.py"
set "RUN_RESULT=%ERRORLEVEL%"
goto :finish

:no_complete_environment
set "ENTRY=%FALLBACK_ENTRY%"
goto :missing_dependencies

:missing_python
echo [ERROR] No working Python found.
echo Install Python 3.9 or newer, or set PERFECTPIXEL_PYTHON to python.exe.
set "RUN_RESULT=1"
goto :finish

:missing_dependencies
echo [ERROR] Required desktop dependencies are missing.
echo Run this command from the project directory:
echo "%ENTRY%" -m pip install numpy opencv-python Pillow PySide6
set "RUN_RESULT=1"
goto :finish

:finish
if not "%~1"=="" exit /b %RUN_RESULT%
echo.
echo [Exit] Press any key to close.
pause >nul
exit /b %RUN_RESULT%

:try_python
if defined ENTRY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
if not defined FALLBACK_ENTRY set "FALLBACK_ENTRY=%~f1"
"%~1" -c "import cv2, numpy; from PIL import Image; from PySide6 import QtCore, QtGui, QtWidgets" >nul 2>&1
if errorlevel 1 goto :skip_incomplete
set "ENTRY=%~f1"
exit /b 0

:skip_incomplete
echo [SKIP] Desktop dependencies unavailable in "%~f1"
exit /b 0
