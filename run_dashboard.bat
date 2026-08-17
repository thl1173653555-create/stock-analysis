@echo off
chcp 65001 >nul
title A股 MA 策略看板
cd /d "%~dp0"
echo 正在启动 A股 MA 策略看板...
echo 启动后浏览器会自动打开: http://localhost:8501
echo 关闭本窗口即停止服务。
echo.
.venv\Scripts\python.exe -m streamlit run dashboard.py
pause
