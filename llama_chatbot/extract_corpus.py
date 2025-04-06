import os
import logging
import pandas as pd
from pinecone import Pinecone
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_corpus():
    """Извлекает корпус из Pinecone и сохраняет его в CSV файл"""
    try:
        # Проверяем API ключ
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY не установлен")
        
        # Инициализация Pinecone
        pc = Pinecone(api_key=api_key)
        index_name = os.environ.get("TFIDF_INDEX_NAME", 'red-llama-tf-idf')
        logger.info(f"Подключение к индексу Pinecone: {index_name}")
        index = pc.Index(index_name)
        
        # Получаем статистику индекса
        stats = index.describe_index_stats()
        logger.info(f"Статистика индекса: {stats}")
        
        # Получаем все вектора через query
        texts = []
        namespace = ""  # Используем пустой namespace
        top_k = 1000  # Максимальное количество векторов за один запрос
        
        # Получаем вектора партиями
        for offset in range(0, stats['total_vector_count'], top_k):
            logger.info(f"Получение векторов с {offset} по {offset + top_k}")
            results = index.query(
                vector=[0] * stats['dimension'],  # Нулевой вектор для получения всех
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
                offset=offset
            )
            
            for match in results.matches:
                if 'metadata' in match and 'chunk_text' in match['metadata']:
                    texts.append(match['metadata']['chunk_text'])
        
        logger.info(f"Извлечено {len(texts)} текстов из метаданных")
        
        if texts:
            # Создаем директорию data, если её нет
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Сохраняем корпус в CSV
            df = pd.DataFrame({'chunk_text': texts})
            csv_path = data_dir / "corpus.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Корпус сохранен в {csv_path}")
        else:
            logger.warning("Не найдено текстов в Pinecone")
            
    except Exception as e:
        logger.error(f"Ошибка при извлечении корпуса: {e}")
        raise

if __name__ == "__main__":
    extract_corpus() 