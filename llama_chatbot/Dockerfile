FROM python:3.11.11-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "healthcheck.py"]

CMD ["sh", "-c", "export $(grep -v '^#' .env | xargs) && streamlit run app.py --server.address=0.0.0.0"]