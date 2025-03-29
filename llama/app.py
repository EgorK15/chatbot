from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import webbrowser
import os
import sys
from config import config_me, path_me
from db import init_db, get_recent_history, save_message, history_limit, history_trim

# Текущие параметры модели
current_temperature = 0.7
tokens_limit = 1000

def log_to_stderr(message):
    print(message, file=sys.stderr)

# Инициализация Flask
app = Flask(__name__)
CORS(app)

# Настройки модели 
API_KEY, api_base, my_model = config_me()

llm = ChatOpenAI(
    base_url=api_base,
    api_key=API_KEY,
    model=my_model,
    temperature=current_temperature,
    max_tokens=tokens_limit,
)

# Инициализация базы данных при запуске
init_db()

# Шаблон для чата
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — дружелюбный помощник Лама. Отвечай на вопросы вежливо и понятно."),
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

        # Формируем входные данные с историей
        formatted_prompt = prompt.format_messages(
            system="Ты — дружелюбный помощник Лама. Отвечай на вопросы вежливо и понятно.",
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
    app.run(debug=True)