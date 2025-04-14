# -*- coding: utf-8 -*-

import os
import pickle
import json
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pandas as pd
import scipy.sparse
from config import PINECONE_API_KEY, TFIDF_INDEX_NAME, DATA_DIR
from tfidf_constants import TFIDF_CONFIG, RELEVANCE_THRESHOLDS

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_tfidf_retriever_instance = None

def get_tfidf_retriever():
    global _tfidf_retriever_instance
    if _tfidf_retriever_instance is None:
        _tfidf_retriever_instance = TFIDFRetriever()
    return _tfidf_retriever_instance

class TFIDFRetriever:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.vectorizer_path = os.path.join(data_dir, "tfidf_vectorizer.pkl")
        self.matrix_path = os.path.join(data_dir, "tfidf_matrix.npz")
        self.metadata_path = os.path.join(data_dir, "tfidf_metadata.json")

        logger.info("Загрузка существующего векторайзера...")
        with open(self.vectorizer_path, "rb") as f:
            self._vectorizer = pickle.load(f)
        logger.info("TF-IDF векторайзер успешно загружен")

        logger.info("Загрузка матрицы TF-IDF...")
        self.tfidf_matrix = scipy.sparse.load_npz(self.matrix_path)
        logger.info("TF-IDF матрица успешно загружена")

        logger.info("Загрузка метаданных корпуса...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata_json = json.load(f)

        logger.info(f"Тип tfidf_metadata.json: {type(metadata_json)}")

        if isinstance(metadata_json, dict):
            if all(isinstance(v, dict) for v in metadata_json.values()):
                logger.warning("tfidf_metadata.json выглядит как словарь словарей — преобразуем в список")
                metadata_json = list(metadata_json.values())
            else:
                raise ValueError("tfidf_metadata.json должен быть списком словарей, но это dict со скалярами")

        if not isinstance(metadata_json, list):
            raise ValueError("tfidf_metadata.json должен быть списком словарей")

        logger.info(f"Пример строки метаданных: {json.dumps(metadata_json[0], ensure_ascii=False)[:200]}")
        self.metadata = pd.DataFrame(metadata_json)
        logger.info("Метаданные успешно загружены")

    @property
    def vectorizer(self):
        return self._vectorizer

    def get_tfidf_vector(self, text: str) -> list:
        try:
            if not hasattr(self.vectorizer, 'idf_'):
                raise ValueError("Векторайзер не обучен")
            return self.vectorizer.transform([text]).toarray()[0].tolist()
        except Exception as e:
            logger.error(f"Ошибка при получении вектора: {e}")
            raise

    def retrieve_local_tfidf_only(self, query_text: str, top_k: int = 50) -> Dict[str, Any]:
        try:
            logger.info("Запускаем локальный TF-IDF поиск...")
            query_vector = self.vectorizer.transform([query_text])
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]
            results = []
            for i in top_indices:
                score = similarities[i]
                row = self.metadata.iloc[i]
                results.append({
                    "score": float(score),
                    "metadata": {
                        "chunk_text": row.get("chunk_text", ""),
                        "topic": row.get("topic", None),
                        "source": row.get("source", ""),
                        "topic_name": row.get("topic_name", ""),
                        "source_type": "tfidf"
                    }
                })
            logger.info(f"Найдено {len(results)} локальных результатов по TF-IDF")
            return {"matches": results}

        except Exception as e:
            logger.error(f"Ошибка при локальном TF-IDF-поиске: {e}")
            return {"matches": []}

    def retrieve(self, query_text: str, context: str = "", topic_code: Optional[int] = None, company_name: Optional[str] = None, top_k: int = 10) -> Dict[str, Any]:
        logger.info("TF-IDF: основной retrieve")
        results = self.retrieve_local_tfidf_only(query_text, top_k=top_k)

        if company_name:
            logger.info(f"Фильтрация по названию компании: {company_name}")
            filtered_matches = [
                m for m in results["matches"]
                if m["metadata"].get("topic_name", "").strip().lower() == company_name.strip().lower()
            ]
            logger.info(f"Оставлено {len(filtered_matches)} из {len(results['matches'])} после фильтрации по topic_name")
            return {"matches": filtered_matches}

        return results
