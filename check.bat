@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo [1/3] Python syntax
"%PY%" -m compileall -q desktop_app.py app.py image_crop.py image_resizer.py image_splitter.py manual_editor.py sequence_preview.py src mcp_server
if errorlevel 1 goto :fail
echo [2/3] MCP smoke test
"%PY%" -m mcp_server.smoke_test
if errorlevel 1 goto :fail
echo [3/3] Desktop construction
set QT_QPA_PLATFORM=offscreen
"%PY%" -c "from PySide6.QtWidgets import QApplication; from desktop_app import MainWindow; app=QApplication([]); w=MainWindow(); print('MAINWINDOW_OK tabs=' + str(w.tabs.count()))"
if errorlevel 1 goto :fail
echo CHECK_OK
exit /b 0
:fail
echo CHECK_FAILED
exit /b 1
