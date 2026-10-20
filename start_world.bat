@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem V3 默认启动：俯视地图世界（世界盒子风格：千人群伙 + 上帝之手）。
rem 想回到旧世界（侧视像素小人）：python -m newworld --classic
python -m newworld %*
if errorlevel 1 pause
