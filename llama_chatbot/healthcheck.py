#!/usr/bin/env python
"""
Простой скрипт для проверки здоровья Streamlit приложения.
Может использоваться в Docker контейнере как HEALTHCHECK.
"""

import requests
import sys

def check_streamlit_health():
    """Проверка доступности Streamlit приложения"""
    try:
        # Пытаемся получить доступ к Streamlit приложению
        response = requests.get("http://localhost:8501", timeout=5)
        
        # Проверяем, что ответ успешный
        if response.status_code == 200:
            print("Streamlit приложение работает нормально.")
            return True
        else:
            print(f"Streamlit приложение вернуло код ошибки: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Ошибка при проверке Streamlit приложения: {e}")
        return False

if __name__ == "__main__":
    # Если проверка успешна, выходим с кодом 0 (успех)
    # В противном случае - с кодом 1 (ошибка)
    sys.exit(0 if check_streamlit_health() else 1) 