
# -*- coding: utf-8 -*-

import sqlite3
import datetime
import os
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем директорию для данных, если её нет
DATA_DIR = os.getenv('DATA_DIR', '/app/data')
os.makedirs(DATA_DIR, exist_ok=True)

# Путь к базе данных в монтируемом volume
DB_NAME = os.path.join(DATA_DIR, "chat_memory.db")

def get_db_connection():
    """Создает и возвращает соединение с базой данных."""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        raise

def create_tables():
    """Creates or updates tables for chat sessions and messages."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Create or update chats table with session metadata.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_name TEXT,
                temperature FLOAT,
                created_at TEXT NOT NULL,
                last_active TEXT,
                status TEXT DEFAULT 'active',
                topic TEXT
            )
        """)

        # Create or update messages table with a status column.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL REFERENCES chats(chat_id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                temperature FLOAT,
                structured_output TEXT,
                status TEXT DEFAULT 'active'
            )
        """)

        conn.commit()
        logger.info("Таблицы успешно созданы или обновлены")
    except sqlite3.Error as e:
        logger.error(f"Ошибка при создании таблиц: {e}")
        raise
    finally:
        conn.close()

def save_chat(chat_id, name, created_at, model_name=None, temperature=None, status='active', topic=None):
    """Save or update a chat session's metadata."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO chats (chat_id, name, created_at, model_name, temperature, status, topic)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (chat_id, name, created_at, model_name, temperature, status, topic))
    conn.commit()
    conn.close()

def get_chats():
    """Retrieve all chat sessions ordered by creation date."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chat_id, name, created_at, last_active, status, model_name, temperature, topic
        FROM chats
        ORDER BY created_at
    """)
    chats = cursor.fetchall()
    conn.close()
    return chats

def update_chat_last_active(chat_id):
    """Update the last active time for a chat session."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.datetime.now().isoformat()
    cursor.execute("UPDATE chats SET last_active = ? WHERE chat_id = ?", (now, chat_id))
    conn.commit()
    conn.close()

def update_chat_name(chat_id, new_name):
    """Update the name of a chat session."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE chats SET name = ? WHERE chat_id = ?", (new_name, chat_id))
    conn.commit()
    conn.close()

def save_message(chat_id, role, content, temperature, structured_output=None):
    """Save a chat message with additional metadata."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (chat_id, role, content, timestamp, temperature, structured_output, status)
        VALUES (?, ?, ?, ?, ?, ?, 'active')
    """, (chat_id, role, content, datetime.datetime.now().isoformat(), temperature, structured_output))
    conn.commit()
    conn.close()

def get_chat_messages(chat_id):
    """Retrieve only active messages ordered by timestamp for a given chat."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, temperature, structured_output
        FROM messages
        WHERE chat_id = ? AND status = 'active'
        ORDER BY timestamp
    """, (chat_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages

def archive_chat(chat_id):
    """Mark a chat session as archived."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE chats SET status = 'archived' WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()

def archive_messages(chat_id):
    """Mark all messages for a given chat as archived."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE messages SET status = 'archived' WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()

def export_data():
    """Экспортирует все данные из базы данных в JSON формат."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Получаем все чаты
        cursor.execute("SELECT * FROM chats")
        chats = [dict(row) for row in cursor.fetchall()]
        
        # Получаем все сообщения
        cursor.execute("SELECT * FROM messages")
        messages = [dict(row) for row in cursor.fetchall()]
        
        data = {
            "chats": chats,
            "messages": messages,
            "exported_at": datetime.datetime.now().isoformat()
        }
        
        return json.dumps(data, ensure_ascii=False, indent=2)
    except sqlite3.Error as e:
        logger.error(f"Ошибка при экспорте данных: {e}")
        raise
    finally:
        conn.close()

def import_data(json_data):
    """Импортирует данные из JSON в базу данных."""
    try:
        data = json.loads(json_data)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Импортируем чаты
        for chat in data["chats"]:
            cursor.execute("""
                INSERT OR REPLACE INTO chats 
                (chat_id, name, model_name, temperature, created_at, last_active, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chat["chat_id"],
                chat["name"],
                chat.get("model_name"),
                chat.get("temperature"),
                chat["created_at"],
                chat.get("last_active"),
                chat.get("status", "active")
            ))
        
        # Импортируем сообщения
        for message in data["messages"]:
            cursor.execute("""
                INSERT OR REPLACE INTO messages 
                (id, chat_id, role, content, timestamp, temperature, structured_output, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message["id"],
                message["chat_id"],
                message["role"],
                message["content"],
                message["timestamp"],
                message.get("temperature"),
                message.get("structured_output"),
                message.get("status", "active")
            ))
        
        conn.commit()
        logger.info("Данные успешно импортированы")
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.error(f"Ошибка при импорте данных: {e}")
        raise
    finally:
        conn.close()

def backup_database():
    """Создает резервную копию базы данных."""
    try:
        backup_path = os.path.join(DATA_DIR, f"chat_memory_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        conn = get_db_connection()
        conn.backup(sqlite3.connect(backup_path))
        logger.info(f"Резервная копия создана: {backup_path}")
        return backup_path
    except sqlite3.Error as e:
        logger.error(f"Ошибка при создании резервной копии: {e}")
        raise
    finally:
        conn.close()

def migrate_database():
    """Добавляет новые колонки в существующие таблицы."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Проверяем существование колонки topic
        cursor.execute("PRAGMA table_info(chats)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'topic' not in columns:
            cursor.execute("ALTER TABLE chats ADD COLUMN topic TEXT")
            logger.info("Добавлена колонка topic в таблицу chats")
        
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Ошибка при миграции базы данных: {e}")
        raise
    finally:
        conn.close()

# Вызываем миграцию при импорте модуля
create_tables()
migrate_database()
