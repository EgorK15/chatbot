# -*- coding: utf-8 -*-

import os
import json
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from config import DATA_DIR
from scipy.sparse import save_npz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METADATA_FILE = os.path.join(DATA_DIR, "corpus_metadata.json")
VECTORIZER_FILE = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")

def load_corpus_metadata() -> dict:
    """Загружает метаданные корпуса"""
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка при загрузке метаданных корпуса: {e}")
        raise

def validate_vectorizer_dimension(vectorizer: TfidfVectorizer, target_dim: int) -> bool:
    """Проверяет соответствие размерности векторайзера"""
    return len(vectorizer.get_feature_names_out()) == target_dim

def train_tfidf(corpus_path: str = os.path.join(DATA_DIR, "corpus.csv")) -> None:
    """Обучение TF-IDF векторайзера на корпусе текстов"""
    try:
        # Проверяем наличие метаданных
        if not os.path.exists(METADATA_FILE):
            raise ValueError("Метаданные корпуса не найдены. Сначала запустите extract_corpus.py")

        # Загружаем метаданные
        metadata = load_corpus_metadata()
        target_dimension = metadata["dimension"]
        logger.info(f"Целевая размерность векторов: {target_dimension}")

        # Загружаем корпус
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Файл корпуса {corpus_path} не найден")

        df = pd.read_csv(corpus_path)
        corpus = df["chunk_text"].tolist()
        logger.info(f"Загружено {len(corpus)} текстов из корпуса")

        # Создаем и обучаем векторайзер
        vectorizer = TfidfVectorizer(
            max_features=target_dimension,  # Устанавливаем нужную размерность
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
            stop_words=['и', 'в', 'на', 'с', 'по', 'для', 'за', 'от', 'к', 'о', 'об'],
            lowercase=True
        )

        vectorizer.fit(corpus)

        # Проверяем размерность
        if not validate_vectorizer_dimension(vectorizer, target_dimension):
            raise ValueError(
                f"Размерность векторайзера ({len(vectorizer.get_feature_names_out())}) "
                f"не соответствует требуемой ({target_dimension})"
            )

        # Сохраняем векторайзер
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(VECTORIZER_FILE, "wb") as f:
            pickle.dump(vectorizer, f)

        logger.info("TF-IDF векторайзер успешно обучен и сохранен")

        # Векторизация корпуса
        tfidf_matrix = vectorizer.transform(corpus)

        # Сохраняем матрицу TF-IDF
        matrix_path = os.path.join(DATA_DIR, "tfidf_matrix.npz")
        save_npz(matrix_path, tfidf_matrix)

        logger.info("TF-IDF векторайзер и матрица успешно сохранены")

    except Exception as e:
        logger.error(f"Ошибка при обучении TF-IDF векторайзера: {e}")
        raise

if __name__ == "__main__":
    train_tfidf()