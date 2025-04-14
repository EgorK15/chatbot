FROM python:3.11.11-slim as builder

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Установка зависимостей в виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Создаем директорию для TF-IDF файлов
RUN mkdir -p /app/tfidf_data

# Копируем скрипты
COPY preload_models.py ./
COPY retrieve.py ./
COPY config.py ./
COPY extract_corpus.py ./
COPY train_tfidf.py ./
COPY credentials.json ./
# Предварительная загрузка моделей


# Финальный этап
FROM python:3.11.11-slim

WORKDIR /app

# Копируем виртуальное окружение и кэш моделей из builder
COPY token.json /app/token.json
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache /root/.cache
COPY --from=builder /app/tfidf_data /app/tfidf_data

ENV PATH="/opt/venv/bin:$PATH"

# Копируем всё приложение
COPY . .

# Копируем и активируем entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "healthcheck.py"]

# Заменяем запуск через streamlit на универсальный entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
