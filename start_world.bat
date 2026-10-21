@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem newworld 实验平台：浏览器观察界面（项目主入口）。
rem 观察优先：界面只做观察与控制；所有命令（暂停/单步/速度/重置）全部入审计。
start "" http://127.0.0.1:8600
python -m newworld.lab_server --port 8600 --seed 0
if errorlevel 1 pause
