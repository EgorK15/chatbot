from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Для обработки CORS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import webbrowser  # Для автоматического открытия браузера
import os

# Инициализация Flask
app = Flask(__name__)
CORS(app)  # Разрешить CORS для всех маршрутов

# Укажите URL вашего сервера с моделью llama3-8b-instruct-8k
api_base = "https://llama3gpu.neuraldeep.tech/v1"
my_model = "llama-3-8b-instruct-8k"
API_KEY = "F9yei26LMobzDdVmdKuEcbIlTod3OoTJXAszIUoMdAwYwGWS50bvKODj99JZRqcJA8mUQgR1zlXoHpCORP97ODLAyfagqrob"

# Настройка модели
llm = ChatOpenAI(
    base_url=api_base,  # URL вашего API Llama 3
    api_key=API_KEY,  # Ваш API-ключ
    model=my_model,  # Название модели
    temperature=0.7,  # Параметр креативности
    max_tokens=1000,  # Максимальное количество токенов в ответе
)

# Шаблон для чата
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — дружелюбный помощник Лама. Отвечай на вопросы вежливо и понятно."),
    ("human", "{message}"),
])

# Создаем цепочку
chain = prompt | llm

# Маршрут для обработки сообщений
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "Сообщение не может быть пустым"}), 400

    try:
        # Получаем ответ от модели
        response = chain.invoke({"message": user_message})
        return jsonify({"response": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Маршрут для отдачи index.html
@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

# Функция для открытия браузера
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')

# Запуск сервера
if __name__ == '__main__':
    # Открываем браузер только при запуске сервера, а не при перезагрузке
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        open_browser()
    app.run(debug=True)