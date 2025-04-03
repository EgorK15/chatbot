
@echo off
REM Переход в директорию, где находится app.py
cd .

REM Активация виртуального окружения (если используется)
REM call venv\Scripts\activate

REM Запуск Flask-приложения в фоновом режиме
start /B python app.py

REM Ожидание нажатия клавиши
pause
