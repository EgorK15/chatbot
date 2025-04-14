
# -*- coding: utf-8 -*-

from langchain.schema import HumanMessage, AIMessage
import datetime
import json
from db import (save_message, get_chat_messages, archive_chat,
               get_chats, save_chat, update_chat_last_active, 
               update_chat_name, archive_messages)

class ChatManager:
    def __init__(self):
        self.chat_sessions = {}
        self.current_chat_id = None
        self._load_chats()
        
        # Если после загрузки чатов нет активных, создаем новый
        if not self.chat_sessions:
            self.create_chat()

    def _load_chats(self):
        """Загружает все активные чаты из базы данных"""
        stored_chats = get_chats()
        if not stored_chats:
            return

        for chat in stored_chats:
            chat_id, name, created_at, last_active, status, model_name_db, temp, topic = chat
            # Пропускаем архивированные чаты
            if status == 'archived':
                continue

            # Загружаем сообщения для чата
            messages = get_chat_messages(chat_id)
            msg_objects = []
            
            for row in messages:
                role, content, message_temperature, structured_output = row
                if role == "user":
                    msg_objects.append(HumanMessage(content=content))
                else:
                    ai_msg = AIMessage(content=content)
                    if structured_output:
                        try:
                            ai_msg.structured_output = json.loads(structured_output)
                        except json.JSONDecodeError:
                            pass
                    msg_objects.append(ai_msg)

            self.chat_sessions[chat_id] = {
                "name": name,
                "created_at": created_at,
                "status": status,
                "messages": msg_objects,
                "model_name": model_name_db,
                "temperature": temp,
                "topic": topic
            }

        # Устанавливаем текущий чат
        if self.chat_sessions:
            self.current_chat_id = list(self.chat_sessions.keys())[0]

    def create_chat(self, name="Новый чат", model_name=None, temperature=None, topic=None):
        """Создает новый чат"""
        from uuid import uuid4
        
        chat_id = str(uuid4())
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        self.chat_sessions[chat_id] = {
            "name": name,
            "messages": [],
            "created_at": created_at,
            "status": "active",
            "model_name": model_name,
            "temperature": temperature,
            "topic": topic
        }
        
        save_chat(chat_id, name, created_at, model_name, temperature, topic=topic)
        self.current_chat_id = chat_id
        return chat_id

    def add_message(self, chat_id, role, content, temperature=None, structured_output=None):
        """Добавляет новое сообщение в чат"""
        if chat_id not in self.chat_sessions:
            return False

        # Создаем объект сообщения
        if role == "user":
            message = HumanMessage(content=content)
        else:
            message = AIMessage(content=content)
            if structured_output:
                message.structured_output = structured_output

        # Добавляем в память
        if "messages" not in self.chat_sessions[chat_id]:
            self.chat_sessions[chat_id]["messages"] = []
        
        self.chat_sessions[chat_id]["messages"].append(message)
        
        # Сохраняем в БД
        save_message(
            chat_id, 
            role, 
            content, 
            temperature,
            json.dumps(structured_output) if structured_output else None
        )
        
        # Обновляем время последней активности
        update_chat_last_active(chat_id)
        return True

    def get_chat_history(self, chat_id):
        """Возвращает историю сообщений чата"""
        if chat_id not in self.chat_sessions:
            return []
        return self.chat_sessions[chat_id]["messages"]

    def rename_chat(self, chat_id, new_name):
        """Переименовывает чат"""
        if chat_id not in self.chat_sessions or not new_name:
            return False
            
        self.chat_sessions[chat_id]["name"] = new_name
        update_chat_name(chat_id, new_name)
        return True

    def archive_chat(self, chat_id):
        """Архивирует чат"""
        if chat_id not in self.chat_sessions:
            return False
            
        archive_messages(chat_id)
        archive_chat(chat_id)
        del self.chat_sessions[chat_id]
        
        # Если есть другие чаты, выбираем первый
        if self.chat_sessions:
            self.current_chat_id = list(self.chat_sessions.keys())[0]
        else:
            self.current_chat_id = self.create_chat()
        return True

    def clear_chat(self, chat_id):
        """Очищает историю сообщений чата"""
        if chat_id not in self.chat_sessions:
            return False
            
        archive_messages(chat_id)
        self.chat_sessions[chat_id]["messages"] = []
        return True

    def get_chat_info(self, chat_id):
        """Возвращает информацию о чате"""
        if chat_id not in self.chat_sessions:
            return None
        return self.chat_sessions[chat_id]

    def get_all_chats(self):
        """Возвращает список всех активных чатов"""
        return {
            chat_id: {
                "name": info["name"],
                "created_at": info["created_at"]
            }
            for chat_id, info in self.chat_sessions.items()
        }

    def update_topic(self, chat_id, topic):
        """Обновляет тему чата"""
        if chat_id not in self.chat_sessions:
            return False
            
        self.chat_sessions[chat_id]["topic"] = topic
        save_chat(
            chat_id,
            self.chat_sessions[chat_id]["name"],
            self.chat_sessions[chat_id]["created_at"],
            self.chat_sessions[chat_id]["model_name"],
            self.chat_sessions[chat_id]["temperature"],
            topic=topic
        )
        return True 
