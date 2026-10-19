@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m newworld --neural %*
if errorlevel 1 pause
