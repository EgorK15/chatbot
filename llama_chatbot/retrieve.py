# -*- coding: utf-8 -*-

import os
from pinecone import Pinecone
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from tfidf_retriever import TFIDFRetriever

# Инициализация Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("INDEX_NAME", 'red-llama-bertopic')
index = pc.Index(index_name)

# Загрузка модели для эмбеддингов
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Инициализация TF-IDF ретривера
tfidf_retriever = TFIDFRetriever()

def get_embedding(text: str) -> list:
    """Получение эмбеддинга текста с помощью sentence-transformers"""
    return model.encode(text).tolist()

def retrieve(query_text: str, context: str = "", topic_code: Optional[int] = None, use_tfidf: bool = True) -> Dict[str, Any]:
    """Поиск релевантных документов с использованием обоих методов"""
    if use_tfidf:
        # Используем TF-IDF с контекстом
        return tfidf_retriever.retrieve(query_text, context, topic_code)
    else:
        # Используем эмбеддинги
        query_embedding = get_embedding(query_text)
        filter_dict = {"topic": topic_code} if topic_code is not None else None
        return index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter=filter_dict
        )

if __name__ == '__main__':
    # Тестовый запрос
    res = retrieve("Бульдог подходит для жизни в частном доме?")
    for match in res["matches"]:
        print(f"Score: {match['score']}")
        print(f"Text: {match['metadata']['chunk_text']}")
        print("---")