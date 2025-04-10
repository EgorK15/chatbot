#!/bin/bash
set -e

echo "📥 Инициализация TF-IDF векторайзера..."
python extract_corpus.py

echo "📊 Обучение TF-IDF..."
python train_tfidf.py

echo "✅ TF-IDF обучение завершено"
echo "🚀 Запуск приложения..."
exec "$@"
