
# -*- coding: utf-8 -*-

import os
import pandas as pd
from pinecone import Pinecone
import logging
from typing import List, Dict, Any
from config import PINECONE_API_KEY, INDEX_NAME, DATA_DIR
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
CORPUS_METADATA_FILE = os.path.join(DATA_DIR, "corpus_metadata.json")
TFIDF_METADATA_FILE = os.path.join(DATA_DIR, "tfidf_metadata.json")

def validate_environment():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY не найден.")
    if not INDEX_NAME:
        raise ValueError("INDEX_NAME не найден.")
    logger.info("Переменные окружения успешно загружены")

def save_corpus_metadata(metadata: Dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CORPUS_METADATA_FILE, 'w') as f:
        json.dump(metadata, f)
    logger.info(f"Метаданные корпуса сохранены в {CORPUS_METADATA_FILE}")

def extract_corpus(output_file: str = os.path.join(DATA_DIR, "corpus.csv")) -> None:
    try:
        validate_environment()

        logger.info("Подключение к Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        stats = index.describe_index_stats()
        logger.info(f"Всего векторов в индексе: {stats.total_vector_count}")
        logger.info(f"Размерность векторов: {stats.dimension}")
        save_corpus_metadata({"dimension": stats.dimension})

        all_ids = []
        for ns in stats.namespaces:
            response = index.query(
                vector=[0.0] * stats.dimension,
                top_k=stats.total_vector_count,
                include_metadata=True,
                vector_id_prefix=ns
            )
            for match in response.matches:
                if hasattr(match, 'id'):
                    all_ids.append(match.id)

        logger.info(f"Получено {len(all_ids)} векторов")

        rows = []
        for i in range(0, len(all_ids), BATCH_SIZE):
            batch_ids = all_ids[i:i + BATCH_SIZE]
            results = index.fetch(ids=batch_ids)
            for vector_id, vector_data in results.vectors.items():
                metadata = getattr(vector_data, 'metadata', {})
                chunk = metadata.get('chunk_text')
                if chunk:
                    rows.append({
                        "chunk_text": chunk,
                        "topic": metadata.get("topic", None),
                        "topic_name": metadata.get("topic_name", ""),  # ✅ добавлено
                        "source": metadata.get("source", "")
                    })
                else:
                    logger.warning(f"Пропущен вектор без chunk_text: {vector_id}")

        df = pd.DataFrame(rows).drop_duplicates(subset=['chunk_text'])

        os.makedirs(DATA_DIR, exist_ok=True)

        df.to_csv(output_file, index=False)
        logger.info(f"Корпус сохранен в {output_file}")

        tfidf_metadata = df.to_dict(orient="records")
        if not isinstance(tfidf_metadata, list):
            raise ValueError("tfidf_metadata должен быть списком словарей")
        logger.info(f"Сохраняем {len(tfidf_metadata)} записей в tfidf_metadata.json")

        with open(TFIDF_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(tfidf_metadata, f, ensure_ascii=False, indent=2)
            if os.path.getsize(TFIDF_METADATA_FILE) == 0:
                raise RuntimeError("tfidf_metadata.json оказался пустым после записи!")
            else:
                logger.info("Проверка пройдена: tfidf_metadata.json не пустой")
        
        logger.info(f"TF-IDF метаданные сохранены в {TFIDF_METADATA_FILE}")

    except Exception as e:
        logger.error(f"Ошибка при извлечении корпуса: {e}")
        raise

if __name__ == "__main__":
    extract_corpus()
