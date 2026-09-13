@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Create .venv and install mcp_server/requirements.txt first. 1>&2
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -u -m mcp_server.server
exit /b %ERRORLEVEL%
