from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import webbrowser
import os
import sys
from config import config_me
from db import init_db, get_recent_history, save_message, history_limit, history_trim
import torch

# Текущие параметры модели
current_temperature = 0.7
tokens_limit = 1000

def log_to_stderr(message):
    print(message, file=sys.stderr)

# Инициализация Flask
app = Flask(__name__)
CORS(app)

# Настройки модели 
API_KEY, api_base, my_model, YOUR_API_KEY, index_name = config_me()


# Глобальные переменные для модели и других компонентов
threshold = 0.17  # Установите желаемый порог

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config import config_me
# Импорт библиотек
import requests, joblib, io
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm

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

loaded_model, index, model = init()
'''
# Загрузка модели перед первым запросом
@app.before_first_request
def load_model():
    global model, topic_model, index
    try:
        model = torch.load("model.pth")  # Загрузка основной модели
        topic_model = torch.load("topic_model.pth")  # Загрузка модели для классификации тем
        index = initialize_pinecone_index()  # Инициализация индекса Pinecone
        print("Модели и индекс успешно загружены")
    except Exception as e:
        print(f"Ошибка при загрузке моделей или индекса: {e}")

# Маршрут для обработки запросов
@app.route("/generate_prompt", methods=["POST"])
def generate_prompt():
    global model, topic_model, index, threshold

    # Проверка, загружены ли модели
    if model is None or topic_model is None or index is None:
        return jsonify({"error": "Модели или индекс не загружены"}), 500

    # Получение данных из запроса
    data = request.json
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "Сообщение пользователя не предоставлено"}), 400

    # Генерация системного запроса
    try:
        system_prompt = make_system_prompt(user_message, topic_model, model, index, threshold)
        return jsonify({"system_prompt": system_prompt})
    except Exception as e:
        return jsonify({"error": f"Ошибка при генерации системного запроса: {e}"}), 500
'''

llm = ChatOpenAI(
    base_url=api_base,
    api_key=API_KEY,
    model=my_model,
    temperature=current_temperature,
    max_tokens=tokens_limit,
)

# Инициализация базы данных при запуске
init_db()

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

def make_system_prompt(user_message, topic_model, model, index, threshold):
    # Классификация запроса пользователя
    topics, probs = topic_model.transform([user_message])
    #doc_topic = int(topics[0]) if probs[0] > threshold else -1
    doc_topic = int(topics[0]) if (probs[0] > threshold).any() else -1
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
# Шаблон для чата
'''
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — дружелюбный помощник Лама. Отвечай на вопросы вежливо и понятно. {system}"),
    ("human", "История диалога:\n{history}\n\nПользователь: {message}"),
])'
'''
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system}"),  # Передаем system_prompt сюда
    ("human", "История диалога:\n{history}\n\nПользователь: {message}"),
])

# Маршрут для обработки сообщений
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_id = data.get('user_id', 'default_user')

    if not user_message:
        return jsonify({"error": "Сообщение не может быть пустым"}), 400

    try:
        # Получаем последние сообещея
        recent_history = get_recent_history(user_id, history_limit-history_trim)
        system_prompt = make_system_prompt(user_message, topic_model, model, index, threshold)

        # Формируем входные данные с историей
        formatted_prompt = prompt.format_messages(
            system=system_prompt,  # Передаем system_prompt
            history="\n".join([f"{msg['role']}: {msg['message']}" for msg in recent_history]),
            message=user_message,
        )

        print()
        print(f"ПАРАМЕТРЫ ПРОМТА. Порог истории: {history_limit-history_trim}")
        print(f"ФОРМАТИРОВАННЫЙ ПРОМТ: {formatted_prompt}")
        # Получаем ответ от модели
        response = llm.invoke(formatted_prompt)

        # Сохраняем сообщение пользователя и ответ модели
        save_message(user_id, "Пользователь", user_message)
        save_message(user_id, "Лама", response.content)
        print()
        print(f"ПАРАМЕТРЫ ОТВЕТА. Температура: {current_temperature}. Токены: {tokens_limit}")
        print(f"ОТВЕТ МОДЕЛИ: response.content.pretty_print: {response.pretty_print}")
        print()
        print(f"ОТВЕТ МОДЕЛИ: response.response_metadata: {response.response_metadata}")
        return jsonify({"response": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Маршрут для получения последних сообщений
@app.route('/history', methods=['GET'])
def history():
    user_id = request.args.get('user_id', 'default_user')
    try:
        return jsonify({"history": get_recent_history(user_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask должен автоматически обслуживть файлы из папки `static` по маршруту `/static/<filename>`

# Маршрут для отдачи статических файлов
@app.route('/images/<filename>')
def serve_image(filename):
    try:
        return send_from_directory('static/images', filename)
    except FileNotFoundError:
        abort(404)

# Маршрут для отдачи index.html
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Маршрут для отдачи styles.css
@app.route('/')
def styles():
    return send_from_directory('static', 'styles.css')

# Функция для открытия браузера
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')
    

# Полное завершение процесса
@app.route('/shutdown', methods=['POST'])
def shutdown():
    os._exit(0)

# Запуск сервера
if __name__ == '__main__':
    # Открываем браузер только при запуске сервера, а не при перезагрузке
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        open_browser()
    app.run(debug=False)
