@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem newworld 实验平台：浏览器观察界面（P2+，项目主入口）。
rem 观察优先：无灾害按钮；实验干预在页面右侧"实验设置"折叠区，全部入审计。
start "" http://127.0.0.1:8600
python -m newworld.lab_server --port 8600 --seed 0
if errorlevel 1 pause
