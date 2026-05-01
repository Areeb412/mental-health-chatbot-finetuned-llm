@echo off
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║       🧠 Mental Health Support Chatbot                     ║
echo ║       Starting Flask Server...                             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

cd /d %~dp0
uv run python app.py

pause
