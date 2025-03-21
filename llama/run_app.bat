@echo off
REM Переход в директорию, где находится app.py
cd /d "F:\ipynb\chatbot\llama"

REM Активация виртуального окружения (если используется)
REM call venv\Scripts\activate

REM Запуск Flask-приложения без консольного окна
start /B pythonw app.py

REM Закрытие терминала
exit
