@echo off
REM Переход в директорию, где находится app.py
cd /d "F:\ipynb\chatbot\llama"

REM Активация виртуального окружения (если используется)
REM call venv\Scripts\activate

REM Запуск Flask-приложения
python app.py

REM Пауза, чтобы окно не закрывалось сразу
pause
