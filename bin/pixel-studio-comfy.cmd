@echo off
rem pixel-studio-comfy: runs the CLI of the repository this file lives in with Python 3.9+ (py launcher, then python).
setlocal
set "PYTHONPATH=%~dp0..;%PYTHONPATH%"
py -3 -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>nul
if errorlevel 1 goto trypython
py -3 -m psc %*
exit /b %ERRORLEVEL%
:trypython
python -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>nul
if errorlevel 1 goto nopython
python -m psc %*
exit /b %ERRORLEVEL%
:nopython
echo error: Python 3.9 or newer was not found (tried py -3 and python). 1>&2
exit /b 1
