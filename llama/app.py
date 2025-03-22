from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import webbrowser
import os
import sqlite3

# Инициализация Flask
app = Flask(__name__)
CORS(app)

# Получаем текущий рабочий каталог
current_dir = os.getcwd()

# Строим путь на три уровня вверх
config_path = os.path.join(current_dir, '..', '..', 'llama.config')

# Нормализуем путь (убираем лишние '..' и т.д.)
config_path = os.path.normpath(config_path)

# Читаем файл
with open(config_path, 'r') as file:
    api_base = file.readline().strip()
    my_model = file.readline().strip()
    API_KEY = file.readline().strip()

llm = ChatOpenAI(
    base_url=api_base,
    api_key=API_KEY,
    model=my_model,
    temperature=0.7,
    max_tokens=1000,
)

# Шаблон для чата
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — дружелюбный помощник Лама. Отвечай на вопросы вежливо и понятно."),
    ("human", "{message}"),
])

# Инициализация базы данных
def init_db():
    with sqlite3.connect('chat_history.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

# Сохранение сообщения
def save_message(user_id, role, message):
    with sqlite3.connect('chat_history.db') as conn:
        conn.execute('''
            INSERT INTO messages (user_id, role, message) VALUES (?, ?, ?)
        ''', (user_id, role, message))
def get_recent_history(user_id):
    with sqlite3.connect('chat_history.db') as conn:
        cursor = conn.execute('''
            SELECT role, message FROM messages 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 8
        ''', (user_id,))
        return [{"role": row[0], "message": row[1]} for row in cursor.fetchall()]


# Инициализация базы данных при запуске
init_db()

# Маршрут для обработки сообщений
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_id = data.get('user_id', 'default_user')

    if not user_message:
        return jsonify({"error": "Сообщение не может быть пустым"}), 400

    try:
        # Получаем последние 8 сообщений
        recent_history = get_recent_history(user_id)

        # Формируем входные данные с историей
        formatted_prompt = prompt.format_messages(
            system="Ты — дружелюбный помощник Лама. Отвечай на вопросы вежливо и понятно.",
            history="\n".join([f"{msg['role']}: {msg['message']}" for msg in recent_history]),
            message=user_message,
        )

        # Получаем ответ от модели
        response = llm.invoke(formatted_prompt)

        # Сохраняем сообщение пользователя и ответ модели
        save_message(user_id, "Пользователь", user_message)
        save_message(user_id, "Лама", response.content)

        return jsonify({"response": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Маршрут для получения последних 8 сообщений
@app.route('/history', methods=['GET'])
def history():
    user_id = request.args.get('user_id', 'default_user')
    try:
        return jsonify({"history": get_recent_history(user_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Маршрут для отдачи index.html
@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')
# Маршрут для отдачи styles.css
@app.route('/')
def styles():
    return send_from_directory(os.path.dirname(__file__), 'styles.css')


# Маршрут для отдачи статических файлов
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

# Маршрут для отдачи favicon.png
@app.route('/favicon.png')
def favicon():
    return send_from_directory('images', 'favicon.png')

# Функция для открытия браузера
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')

# Запуск сервера
if __name__ == '__main__':
    # Открываем браузер только при запуске сервера, а не при перезагрузке
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        open_browser()
    app.run(debug=True)