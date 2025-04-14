#!/bin/bash
set -e

echo "📥 Проверка TF-IDF корпуса..."
if [ ! -f "./tfidf_data/tfidf_metadata.json" ]; then
  echo "📦 Извлечение корпуса..."
  python extract_corpus.py

  echo "📊 Обучение TF-IDF..."
  python train_tfidf.py

  echo "✅ TF-IDF обучение завершено"
else
  echo "✅ TF-IDF уже существует, пропускаем обучение."
fi

echo "📥 Предзагрузка модели..."
python preload_models.py || echo "⚠️ Модель не загрузилась на этом этапе"


echo "🚀 Запуск приложения..."
exec "$@"


