import os
import pickle
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from pinecone import Pinecone
import logging
import pandas as pd
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFIDFRetriever:
    def __init__(self):
        # Инициализация Pinecone
        try:
            api_key = os.environ.get("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY не установлен")
            
            self.pc = Pinecone(api_key=api_key)
            self.index_name = os.environ.get("TFIDF_INDEX_NAME", 'red-llama-tf-idf')
            logger.info(f"Подключение к индексу Pinecone: {self.index_name}")
            self.index = self.pc.Index(self.index_name)
            
            # Загрузка или обучение векторайзера
            self.vectorizer = self._initialize_vectorizer()
        except Exception as e:
            logger.error(f"Ошибка при инициализации TFIDFRetriever: {e}")
            raise
        
    def _extract_corpus_from_pinecone(self) -> list:
        """Извлечение корпуса из Pinecone"""
        try:
            logger.info("Извлечение корпуса из Pinecone...")
            
            # Проверяем подключение к Pinecone
            if not hasattr(self, 'index'):
                raise ValueError("Не удалось подключиться к Pinecone")
            
            # Получаем статистику индекса
            stats = self.index.describe_index_stats()
            logger.info(f"Статистика индекса: {stats}")
            
            # Получаем все вектора
            results = self.index.fetch(ids=[])
            logger.info(f"Получено {len(results.vectors)} векторов")
            
            texts = []
            for vector_id, vector_data in results.vectors.items():
                if 'metadata' in vector_data and 'chunk_text' in vector_data['metadata']:
                    texts.append(vector_data['metadata']['chunk_text'])
            
            logger.info(f"Извлечено {len(texts)} текстов из метаданных")
            
            if texts:
                # Сохраняем корпус в CSV для будущего использования
                df = pd.DataFrame({'chunk_text': texts})
                df.to_csv('corpus.csv', index=False)
                logger.info(f"Создан corpus.csv с {len(texts)} текстами")
                return texts
            else:
                logger.warning("Не найдено текстов в Pinecone")
                return []
        except Exception as e:
            logger.error(f"Ошибка при извлечении корпуса из Pinecone: {e}")
            return []
        
    def _initialize_vectorizer(self) -> TfidfVectorizer:
        """Инициализация или загрузка векторайзера"""
        try:
            # Пробуем загрузить существующий векторайзер
            if os.path.exists("tfidf_vectorizer.pkl"):
                logger.info("Попытка загрузки существующего векторайзера...")
                with open("tfidf_vectorizer.pkl", "rb") as f:
                    vectorizer = pickle.load(f)
                    if hasattr(vectorizer, 'idf_'):
                        logger.info("TF-IDF векторайзер успешно загружен")
                        return vectorizer
                    else:
                        logger.warning("Загруженный векторайзер не обучен")
        except Exception as e:
            logger.warning(f"Ошибка при загрузке векторайзера: {e}")
        
        # Если векторайзер не загружен или не обучен, создаем новый
        logger.info("Создание нового векторайзера...")
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
            stop_words=['и', 'в', 'на', 'с', 'по', 'для', 'за', 'от', 'к', 'о', 'об'],
            lowercase=True
        )
        
        # Пробуем найти и загрузить корпус для обучения
        corpus_paths = [
            "data/corpus.csv",
            "corpus.csv",
            "../data/corpus.csv"
        ]
        
        corpus = None
        for path in corpus_paths:
            try:
                if os.path.exists(path):
                    logger.info(f"Попытка загрузки корпуса из {path}")
                    df = pd.read_csv(path)
                    if "chunk_text" in df.columns:
                        corpus = df["chunk_text"].tolist()
                        logger.info(f"Корпус загружен из {path}, {len(corpus)} текстов")
                        logger.info(f"Пример текста из корпуса: {corpus[0][:100]}...")
                        break
                    else:
                        logger.warning(f"Файл {path} не содержит колонку 'chunk_text'")
            except Exception as e:
                logger.warning(f"Ошибка при загрузке корпуса из {path}: {e}")
        
        if corpus and len(corpus) > 0:
            # Обучаем векторайзер
            logger.info(f"Обучение векторайзера на {len(corpus)} текстах...")
            try:
                vectorizer.fit(corpus)
                logger.info("TF-IDF векторайзер успешно обучен")
                
                # Проверяем, что векторайзер действительно обучен
                if not hasattr(vectorizer, 'idf_'):
                    raise ValueError("Векторайзер не был обучен корректно")
                
                # Сохраняем обученный векторайзер
                try:
                    with open("tfidf_vectorizer.pkl", "wb") as f:
                        pickle.dump(vectorizer, f)
                    logger.info("TF-IDF векторайзер сохранен")
                except Exception as e:
                    logger.error(f"Ошибка при сохранении векторайзера: {e}")
            except Exception as e:
                logger.error(f"Ошибка при обучении векторайзера: {e}")
                raise
        else:
            error_msg = "Корпус для обучения не найден или пуст. Пожалуйста, запустите extract_corpus.py для создания корпуса"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return vectorizer

    def get_tfidf_vector(self, text: str) -> list:
        """Получение TF-IDF вектора для текста"""
        try:
            if not hasattr(self.vectorizer, 'idf_'):
                raise ValueError("Векторайзер не обучен")
            return self.vectorizer.transform([text]).toarray()[0].tolist()
        except Exception as e:
            logger.error(f"Ошибка при получении вектора: {e}")
            raise

    def retrieve(self, query_text: str, context: str, topic_code: Optional[int] = None) -> Dict[str, Any]:
        """Поиск релевантных документов с использованием TF-IDF"""
        try:
            # Проверяем, обучен ли векторайзер
            if not hasattr(self.vectorizer, 'idf_'):
                raise ValueError("TF-IDF векторайзер не обучен. Пожалуйста, убедитесь, что файл корпуса (corpus.csv) доступен.")
            
            # Объединяем контекст и запрос
            combined_text = f"{context} {query_text}"
            
            # Получаем TF-IDF вектор
            query_vector = self.get_tfidf_vector(combined_text)
            
            # Фильтр по теме, если указан
            filter_dict = {"topic": topic_code} if topic_code is not None else None
            
            # Поиск в Pinecone
            logger.info(f"Выполняем поиск с фильтром: {filter_dict}")
            logger.info(f"Запрос: {combined_text}")
            logger.info(f"Размер вектора запроса: {len(query_vector)}")
            
            results = self.index.query(
                vector=query_vector,
                top_k=20,  # Увеличиваем количество результатов
                include_metadata=True,
                filter=filter_dict
            )
            
            logger.info(f"Найдено {len(results['matches'])} результатов")
            for i, match in enumerate(results['matches']):
                logger.info(f"Результат {i+1}: score={match['score']:.4f}")
                logger.info(f"Текст: {match['metadata']['chunk_text'][:200]}...")
                if 'topic' in match['metadata']:
                    logger.info(f"Тема: {match['metadata']['topic']}")
            
            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise 