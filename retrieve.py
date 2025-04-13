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
    global _model
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return _model

def get_pinecone_index():
    global _pinecone, _index
    if _index is None:
        if _pinecone is None:
            _pinecone = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pinecone.Index(INDEX_NAME)
    return _index

def get_embedding(text: str) -> list:
    model = get_embedding_model()
    return model.encode(text).tolist()

def is_relevant(search_results: Dict[str, Any], use_tfidf: bool = True) -> bool:
    if not search_results.get("matches"):
        return False
    max_score = max(match["score"] for match in search_results["matches"])
    threshold = 0.1 if use_tfidf else 0.13
    return max_score >= threshold

def retrieve(query_text: str, context: str = "", topic_code: Optional[int] = None, use_tfidf: bool = True) -> Dict[str, Any]:
    use_tfidf = topic_code == -2

    if use_tfidf:
        known_companies = {
            "Rectifier Technologies Ltd": ["rectifier technologies", "rectifier"],
            "Brave Bison": ["brave bison", "bison"],
            "Starvest plc": ["starvest", "starvest plc"]
        }
        query_lower = query_text.lower()
        company_name = next(
            (name for name, aliases in known_companies.items() if any(alias in query_lower for alias in aliases)),
            None
        )
        logger.info(f"Определено название компании из запроса: {company_name}")

        if not company_name:
            return {"matches": []}

        query_vector = get_embedding(query_text)
        index = get_pinecone_index()

        pinecone_matches = index.query(
            vector=query_vector,
            top_k=50,
            include_metadata=True,
            filter={"topic": -2}
        ).matches

        filtered_pinecone = [
            {
                "score": float(match.score),
                "metadata": {
                    **match.metadata,
                    "source_type": "pinecone"
                }
            }
            for match in pinecone_matches
            if match.metadata.get("topic_name", "").strip().lower() == company_name.strip().lower()
        ]

        filtered_pinecone.sort(key=lambda x: x["score"], reverse=True)
        top_pinecone = filtered_pinecone[:20]

        tfidf_results = get_tfidf_retriever().retrieve(
            query_text=query_text,
            topic_code=-2,
            company_name=company_name,
            top_k=50
        )

        tfidf_matches = tfidf_results.get("matches", [])
        tfidf_matches.sort(key=lambda x: x["score"], reverse=True)
        top_tfidf = tfidf_matches[:10]

        combined = top_pinecone + top_tfidf
        logger.info(f"Комбинированный результат: {len(combined)} документов")
        return {"matches": combined}

    else:
        query_embedding = get_embedding(query_text)
        filter_dict = {}
        if topic_code is not None:
            filter_dict["topic"] = topic_code
        logger.info(f"Применяем фильтр: {filter_dict}")
        index = get_pinecone_index()
        return index.query(
            vector=query_embedding,
            top_k=50,
            include_metadata=True,
            filter=filter_dict
        )

def debug_index_contents():
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    logger.info(f"Статистика индекса: {stats}")
    for topic in range(-2, 10):
        results = index.query(
            vector=[0.0] * stats.dimension,
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
    debug_index_contents()
    res = retrieve("Бульдог подходит для жизни в частном доме?")
    for match in res["matches"]:
        print(f"Score: {match['score']}")
        print(f"Text: {match['metadata']['chunk_text']}")
        print("---")

