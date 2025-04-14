from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preload_models():
    """Предварительная загрузка моделей"""
    logger.info("Начинаю загрузку sentence-transformers модели...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # Делаем тестовый прогон для проверки
    _ = model.encode("Тестовое предложение")
    logger.info("Модель sentence-transformers успешно загружена")

if __name__ == "__main__":
    preload_models()