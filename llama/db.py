import sqlite3
from config import path_me

# Текущие параметры истории и памяти
history_limit = 34
history_trim = 10

# БД на три уровня вверх
db_path = path_me('chat_history.db')

# Инициализация базы данных
def init_db():
    with sqlite3.connect(db_path) as conn:
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
    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            INSERT INTO messages (user_id, role, message) VALUES (?, ?, ?)
        ''', (user_id, role, message))

def get_recent_history(user_id, max_messages=history_limit):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute('''
            SELECT role, message 
            FROM (
                SELECT role, message, timestamp 
                FROM messages 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ) 
            ORDER BY timestamp ASC
        ''', (user_id, max_messages,))
        return [{"role": row[0], "message": row[1]} for row in cursor.fetchall()]