FROM python:3.11.11-slim

# Рабочая директория
WORKDIR /app

# Копируем Pipfile и Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Устанавливаем pipenv и зависимости проекта
RUN pip install --upgrade pip && \
    pip install pipenv && \
    pipenv install --system --deploy

# Дополнительные библиотеки, если их нет в Pipfile
RUN pip install requests joblib bertopic sentence-transformers pinecone-client python-dotenv numpy

# Копируем весь проект
COPY . .

# Открытие порта для Streamlit
EXPOSE 8501

# Проверка здоровья приложения (если используется healthcheck.py)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "healthcheck.py"]

# Запуск приложения Streamlit напрямую (без pipenv run, так как зависимости уже установлены system-wide)
CMD ["sh", "-c", "export $(grep -v '^#' .env | xargs) && streamlit run app.py --server.address=0.0.0.0"]
