# -*- coding: utf-8 -*-

from pinecone import Pinecone
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from tfidf_retriever import TFIDFRetriever, get_tfidf_retriever
from config import PINECONE_API_KEY, INDEX_NAME
import logging

# Приватные переменные для синглтонов
_model = None
_pinecone = None
_index = None

logger = logging.getLogger(__name__)

def get_embedding_model():
    """Ленивая инициализация и получение модели эмбеддингов"""
    global _model
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return _model

def get_pinecone_index():
    """Ленивая инициализация и получение индекса Pinecone"""
    global _pinecone, _index
    if _index is None:
        if _pinecone is None:
            _pinecone = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pinecone.Index(INDEX_NAME)
    return _index

def get_embedding(text: str) -> list:
    """Получение эмбеддинга текста с помощью sentence-transformers"""
    model = get_embedding_model()
    return model.encode(text).tolist()

def is_relevant(search_results: Dict[str, Any], use_tfidf: bool = True) -> bool:
    """Проверяет, являются ли результаты поиска релевантными
    
    Args:
        search_results: Результаты поиска от Pinecone или TF-IDF
        use_tfidf: Флаг, указывающий использовался ли TF-IDF поиск
    
    Returns:
        bool: True если результаты релевантны, False в противном случае
    """
    if not search_results.get("matches"):
        return False
        
    max_score = max(match["score"] for match in search_results["matches"])
    threshold = 0.1 if use_tfidf else 0.13
    
    return max_score >= threshold

def retrieve(query_text: str, context: str = "", topic_code: Optional[int] = None, use_tfidf: bool = True) -> Dict[str, Any]:
    """Поиск релевантных документов"""
    if use_tfidf:
        # Используем TF-IDF
        retriever = get_tfidf_retriever()
        return retriever.retrieve(query_text, context, topic_code)
    else:
        # Используем эмбеддинги
        query_embedding = get_embedding(query_text)
        
        # Формируем фильтр только по теме
        filter_dict = {}
        if topic_code is not None:
            filter_dict["topic"] = topic_code
            
        logger.info(f"Применяем фильтр: {filter_dict}")
        
        index = get_pinecone_index()
        return index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter=filter_dict
        )

def debug_index_contents():
    """Отладочная функция для проверки содержимого индекса"""
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    logger.info(f"Статистика индекса: {stats}")
    
    # Проверяем все темы
    for topic in range(-2, 10):  # предполагаемый диапазон тем
        results = index.query(
            vector=[0.0] * stats.dimension,  # нулевой вектор для получения всех документов
            top_k=5,
            filter={"topic": topic},
            include_metadata=True
        )
        if results.matches:
            logger.info(f"\nТема {topic}:")
            for match in results.matches:
                logger.info(f"ID: {match.id}")
                logger.info(f"Текст: {match.metadata.get('chunk_text', '')[:200]}...")

if __name__ == '__main__':
    # Добавляем вызов отладочной функции
    debug_index_contents()
    
    # Существующий тестовый код
    res = retrieve("Бульдог подходит для жизни в частном доме?")
    for match in res["matches"]:
        print(f"Score: {match['score']}")
        print(f"Text: {match['metadata']['chunk_text']}")
        print("---")