import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config import config_me, path_me
# Импорт библиотек
import requests, joblib, io
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm
#from app import model, topic_model, threshold


API_KEY, api_base, my_model, YOUR_API_KEY, index_name = config_me()

def init():
    # URL вашей модели
    model_url = 'https://storage.yandexcloud.net/cyrilos/red_bertopic_model'

    # Прогресс-бар для загрузки модели
    print("Загрузка модели...")
    response = requests.get(model_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 КБ
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    # Загрузка модели из байтового потока
    loaded_model_bytes = io.BytesIO()
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        loaded_model_bytes.write(data)
    progress_bar.close()

    # Проверка успешности загрузки
    if response.status_code == 200:
        loaded_model = joblib.load(loaded_model_bytes)
        print("Модель успешно загружена в память.")
    else:
        print(f"Ошибка загрузки: {response.status_code}")
        return None, None, None

    # Прогресс-бар для инициализации Pinecone
    print("Инициализация Pinecone...")
    pc = Pinecone(api_key=YOUR_API_KEY)  # Замените на ваш API-ключ
    index_name = "red-llama-bertopic"
    index = pc.Index(index_name)

    # Прогресс-бар для загрузки модели векторизации
    print("Загрузка модели для векторизации...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    return loaded_model, index, model




# Функция для поиска в Pinecone
def find_in_pinecone(query_text, doc_topic, model, index):
    # Фильтр для поиска в Pinecone
    filter_condition = {
        "topic": {
            "$eq": doc_topic
        }
    }


    # Векторизация запроса
    query_embedding = model.encode(query_text)

    # Поиск в индексе
    results = index.query(
        vector=query_embedding.tolist(),  # Вектор запроса
        top_k=5,  # Количество возвращаемых результатов
        include_metadata=True,  # Включить метаданные в результаты
        filter=filter_condition  # Условия фильтрации
    )

    # Возвращаем результаты
    return results





'''
    # Пример данных для предсказания
    docs = [
        "Как работает машинное обучение?",
        "Какие есть книги по Python?",
        "Как настроить сервер на Linux?"
    ]

    # Использование модели для предсказания тем
    topics, probs = loaded_model.transform(docs)

    # Создание DataFrame для хранения результатов
    docs_df = pd.DataFrame(docs, columns=['document'])

    # Установка порога вероятности
    threshold = 0.17  # Установите желаемый порог

    # Обработка вероятностей и определение тем
    if probs.ndim > 1:  # Если probs двумерный
        print(f'Многомерный: {len(probs)}')
        # Создаем DataFrame для вероятностей
        probs_df = pd.DataFrame(probs, columns=[f'Topic_{i}' for i in range(probs.shape[1])])
        
        # Определяем темы на основе порога
        max_probs = probs_df.max(axis=1)  # Находим максимальные вероятности
        topics = np.where(max_probs > threshold, probs_df.idxmax(axis=1).str.replace('Topic_', '').astype(int), -1)
        
        # Объединяем два DataFrame
        docs_df = pd.concat([docs_df, probs_df], axis=1)
    else:  # Если probs одномерный
        docs_df['probs'] = probs
        docs_df['topic'] = np.where(probs > threshold, -1, topics)  # Присваиваем -1, если вероятность ниже порога

    # Добавляем темы в DataFrame
    docs_df['topic'] = topics
'''

# Функция для поиска в Pinecone
def find_in_pinecone(query_text, doc_topic, model, index):
    # Фильтр для поиска в Pinecone
    filter_condition = {
        "topic": {
            "$eq": doc_topic
        }
    }


    # Векторизация запроса
    query_embedding = model.encode(query_text)

    # Поиск в индексе
    results = index.query(
        vector=query_embedding.tolist(),  # Вектор запроса
        top_k=5,  # Количество возвращаемых результатов
        include_metadata=True,  # Включить метаданные в результаты
        filter=filter_condition  # Условия фильтрации
    )
    print('find_in_pinecone')
    # Возвращаем результаты
    return results

### 2. **Интеграция результатов в запрос к чату**


# Функция для формирования системного запроса
def make_system_prompt(user_message, topic_model, model, index, threshold=0.5):
    """
    Формирует системный запрос на основе пользовательского сообщения и результатов поиска.

    :param user_message: Сообщение пользователя
    :param topic_model: Модель для классификации тем
    :param model: Модель для поиска
    :param index: Индекс для поиска в Pinecone
    :param threshold: Пороговое значение для классификации темы
    :return: Сформированный системный запрос
    """
    # Классификация запроса пользователя
    topics, probs = topic_model.transform([user_message])
    
    # Определение темы документа на основе вероятностей
    doc_topic = int(topics[0]) if probs[0].max() > threshold else -1
    
    # Поиск в Pinecone
    results = find_in_pinecone(user_message, doc_topic, model, index)

    # Формирование контекста для запроса к модели
    context = "Результаты поиска:\n"
    if results['matches']:
        for match in results['matches']:
            if 'answer' in match['metadata']:
                context += f"- {match['metadata']['answer']}\n"
    else:
        context += "По вашему запросу ничего не найдено.\n"

    # Формирование системного запроса
    system_prompt = f"""
    Вот контекст, который может быть полезен для ответа на вопрос пользователя:
    {context}
    """
    return system_prompt


# Вспомогательная функция для инициализации Pinecone
def initialize_pinecone_index():
    """
    Инициализирует индекс Pinecone.
    """
    # Здесь должен быть код для инициализации индекса Pinecone
    return "pinecone_index"  # Заглушка

# Вспомогательная функция для поиска в Pinecone
def find_in_pinecone_1(query, doc_topic, model, index):
    """
    Выполняет поиск в Pinecone на основе запроса и темы.

    :param query: Запрос пользователя
    :param doc_topic: Тема документа
    :param model: Модель для поиска
    :param index: Индекс Pinecone
    :return: Результаты поиска
    """
    # Здесь должен быть код для поиска в
    # Здесь должен быть код для поиска в Pinecone
    # Пример заглушки для возврата результатов
    if not results['matches']:
        print("По вашему запросу ничего не найдено.")
    else:
        print("\nРезультаты поиска в Pinecone:")
        for match in results['matches']:
            print(f"ID: {match['id']}, Score: {match['score']}")
            if 'answer' in match['metadata']:
                print(f"Ответ: {match['metadata']['answer']}")
            if 'topic_name' in match['metadata']:
                print(f"Тема: {match['metadata']['topic_name']}")

    return {
        "matches": [
            {
                "id": "1",
                "score": 0.95,
                "metadata": {
                    "answer": "Машинное обучение — это область искусственного интеллекта, которая позволяет компьютерам обучаться на данных."
                }
            },
            {
                "id": "2",
                "score": 0.90,
                "metadata": {
                    "answer": "Машинное обучение включает в себя алгоритмы, такие как линейная регрессия, деревья решений и нейронные сети."
                }
            }
        ]
    }
