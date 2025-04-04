# -*- coding: utf-8 -*-

import sqlite3
import datetime
import os

os.makedirs('/app/data', exist_ok=True)

# Путь к базе данных в монтируемом volume
DB_NAME = "/app/data/chat_memory.db"


def create_tables():
    """Creates or updates tables for chat sessions and messages."""
    conn = sqlite3.connect(DB_NAME)
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
            status TEXT DEFAULT 'active'
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
    conn.close()

def save_chat(chat_id, name, created_at, model_name=None, temperature=None, status='active'):
    """Save or update a chat session's metadata."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO chats (chat_id, name, created_at, model_name, temperature, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (chat_id, name, created_at, model_name, temperature, status))
    conn.commit()
    conn.close()

def get_chats():
    """Retrieve all chat sessions ordered by creation date."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chat_id, name, created_at, last_active, status, model_name, temperature
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