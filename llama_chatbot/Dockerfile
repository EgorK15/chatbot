FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY Pipfile Pipfile.lock* ./
RUN pip install pipenv && \
    pipenv install --deploy --system && \
    pip uninstall -y pipenv && \
    pip install requests

# Копирование файлов проекта
COPY . .

# Открытие порта для Streamlit
EXPOSE 8501

# Проверка здоровья приложения
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "healthcheck.py"]

# Запуск приложения
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"] 