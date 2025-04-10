# -*- coding: utf-8 -*-

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
import json
import re
import uuid
import datetime
import logging
from config import (
    OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME, 
    TEMPERATURE, PINECONE_API_KEY, INDEX_NAME
)
from db import (create_tables, save_message, get_chat_messages, archive_chat,
                get_chats, save_chat, update_chat_last_active, update_chat_name,
                archive_messages)
from retrieve import retrieve as retrieve_docs, is_relevant
from tfidf_retriever import get_tfidf_retriever
from sentence_transformers import SentenceTransformer
from routing import detect_topic_combined
from chat_manager import ChatManager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализируем таблицы базы данных
create_tables()

# Определение структуры вывода с помощью Pydantic с дополнительным полем note
class ChatResponse(BaseModel):
    content: str = Field(description="Ответ на вопрос пользователя")
    sources: list = Field(description="Источники информации, использованные для ответа", default_factory=list)
    confidence: float = Field(description="Уровень уверенности в ответе от 0 до 1", ge=0, le=1)
    note: str = Field(description="Примечание об источниках, если необходимо", default="")

# Настройка заголовка приложения
st.title('Llama Chatbot')

# Инициализация ChatManager
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()

# Боковая панель для настроек
with st.sidebar:
    st.header("Чаты")
    # Получаем список чатов из ChatManager
    chat_options = st.session_state.chat_manager.get_all_chats()

    if chat_options:
        selected_chat = st.selectbox(
            "Выберите чат",
            options=list(chat_options.keys()),
            format_func=lambda x: f"{chat_options[x]['name']} ({chat_options[x]['created_at']})",
            index=list(chat_options.keys()).index(st.session_state.chat_manager.current_chat_id)
        )
        if selected_chat != st.session_state.chat_manager.current_chat_id:
            st.session_state.chat_manager.current_chat_id = selected_chat
            st.rerun()

    # Настраиваем колонки с разной шириной
    col1, col2, col3 = st.columns([0.9, 1, 1.1])  # Первая колонка 90%, вторая 100%, третья 110%
    with col1:
        if st.button("Новый чат"):
            st.session_state.chat_manager.create_chat()
            st.rerun()
    with col2:
        if st.button("Удалить чат"):
            st.session_state.chat_manager.archive_chat(st.session_state.chat_manager.current_chat_id)
            st.rerun()
    with col3:
        if st.button("Очистить чат"):
            st.session_state.chat_manager.clear_chat(st.session_state.chat_manager.current_chat_id)
            st.rerun()

    # Получаем информацию о текущем чате
    current_chat = st.session_state.chat_manager.get_chat_info(st.session_state.chat_manager.current_chat_id)
    new_chat_name = st.text_input(
        "Переименовать чат",
        value=current_chat["name"] if current_chat else "Новый чат"
    )
    if current_chat and new_chat_name != current_chat["name"]:
        st.session_state.chat_manager.rename_chat(st.session_state.chat_manager.current_chat_id, new_chat_name)

    st.divider()
    st.header("Настройки API")
    api_key = st.text_input("API ключ", value=OPENAI_API_KEY, type="password")
    api_base = st.text_input("API URL", value=OPENAI_API_BASE)
    model_name = st.text_input("Название модели", value=MODEL_NAME)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=TEMPERATURE, step=0.1)

    st.divider()
    st.header("Настройки Pinecone DB")
    pinecone_db_name = st.text_input("Имя Pinecone DB", value=INDEX_NAME)
    pinecone_api_key = st.text_input("Pinecone API ключ", value=PINECONE_API_KEY, type="password")

# Конфигурация клиента LLM
api_changed = (
    "llm" not in st.session_state or
    st.session_state.get("api_key", "") != api_key or
    st.session_state.get("api_base", "") != api_base or
    st.session_state.get("model_name", "") != model_name or
    st.session_state.get("temperature", 0) != temperature
)
if api_changed:
    st.session_state.api_key = api_key
    st.session_state.api_base = api_base
    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["MODEL_NAME"] = model_name
    os.environ["INDEX_NAME"] = pinecone_db_name
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    try:
        st.session_state.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=api_base
        )
        if "embedding_model" not in st.session_state:
            st.session_state.embedding_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        st.sidebar.success("API настроен успешно!")
    except Exception as e:
        st.sidebar.error(f"Ошибка настройки API: {str(e)}")

# После получения истории сообщений
current_messages = st.session_state.chat_manager.get_chat_history(st.session_state.chat_manager.current_chat_id)

for message in current_messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            if hasattr(message, 'structured_output') and message.structured_output:
                st.write(message.content)
                with st.expander("Структурированный ответ"):
                    st.json(message.structured_output)
            else:
                st.write(message.content)

# Обработка нового сообщения пользователя
user_input = st.chat_input("Введите ваше сообщение...")
if user_input:
    # Сначала проверяем, является ли сообщение простым приветствием
    greeting_patterns = [
        r'^\s*(привет|здравствуй|добрый день|доброе утро|добрый вечер|хай|хей|йоу)\s*$',
        r'^\s*(hi|hello|hey|good morning|good afternoon|good evening)\s*$'
    ]
    
    if any(re.match(pattern, user_input.lower()) for pattern in greeting_patterns):
        # Для приветствий пропускаем сложную обработку
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            greeting_response = "Здравствуйте! Чем могу помочь?"
            st.write(greeting_response)
            st.session_state.chat_manager.add_message(
                st.session_state.chat_manager.current_chat_id,
                "assistant",
                greeting_response,
                temperature
            )
            st.stop()

    # Для остальных сообщений продолжаем обычную обработку
    # Отображаем запрос пользователя
    with st.chat_message("user"):
        st.write(user_input)
        # Сохраняем сообщение пользователя в историю
        st.session_state.chat_manager.add_message(
            st.session_state.chat_manager.current_chat_id,
            "user",
            user_input,
            temperature
        )

    # Определяем тему запроса
    topic_result = detect_topic_combined(user_input)
    logger.info(f"Определённая тема: {topic_result['topic_name']} "
                f"(код {topic_result['topic']}, уверенность: {topic_result['confidence']:.2f})\n"
                f"Причина: {topic_result['reasoning']}")

    # Объединенный промпт для переформулировки и определения компании
    combined_prompt = """Выполни следующие шаги:

1. Переформулируй запрос, убрав все лишние слова и сделав его более четким.
2. Определи название компании из списка:
   - Brave Bison
   - Rectifier Technologies Ltd
   - Starvest plc
   
Если в запросе упоминается "Rectifier Technologies" или "Rectifier", считай это как "Rectifier Technologies Ltd".
Если компания не соответствует списку точно, укажи "unknown".

Формат ответа (строго JSON):
{
    "rephrased": "переформулированный запрос",
    "company": "название компании или unknown"
}

Пример ответа:
{
    "rephrased": "revenue of Rectifier Technologies",
    "company": "Rectifier Technologies Ltd"
}

Запрос: {query}"""

    # Поиск релевантной информации
    with st.spinner(""):
        try:
            retriever = get_tfidf_retriever()
            use_tfidf = topic_result["topic"] == -2

            # Перевод запроса (если тема -2)
            if use_tfidf:
                translation_response = st.session_state.llm.invoke(
                    f"Переведи на английский: {user_input}"
                )
                final_query = translation_response.content
                logger.info(f"Переведенный запрос: {final_query}")
            else:
                final_query = user_input

            query_text = f"{final_query}"
            logger.info(f"Поисковый запрос: {query_text}")

            # Используем либо локальный TF-IDF поиск, либо Pinecone
            if use_tfidf:
                res = retriever.retrieve_local_tfidf_only(query_text)
            else:
                res = retriever.retrieve(
                    query_text=query_text,
                    context="",
                    topic_code=topic_result["topic"],
                    embedding_model=st.session_state.embedding_model
                )
            
            # Логирование результатов
            for match in res["matches"]:
                logger.info(f"[RAG MATCH] Score: {match['score']:.4f}")
                logger.info(f"[RAG TEXT] {match['metadata']['chunk_text'][:200]}")
                logger.info(f"[RAG METADATA] topic_name: {match['metadata'].get('topic_name', 'не указано')}")

            # Проверка релевантности найденных документов
            if not is_relevant(res, use_tfidf):
                logger.info("Релевантные документы не найдены, генерирую ответ через LLM")
                response = st.session_state.llm.invoke(user_input)
                with st.chat_message("assistant"):
                    st.write(response.content)
                st.session_state.chat_manager.add_message(
                    st.session_state.chat_manager.current_chat_id,
                    "assistant",
                    response.content,
                    temperature
                )
                st.stop()

            # Формирование ответа на основе найденных документов
            sources = [match["metadata"]["chunk_text"] for match in res["matches"][:5]]
            sources_text = "\n".join(f"Источник {i+1}:\n{source}" for i, source in enumerate(sources))

            # Специальный промпт для финансовых отчетов
            if topic_result["topic"] == -2:
                company_name = res["matches"][0]["metadata"].get("topic_name", "")
                prompt = f"""Ты финансовый аналитик. Ответь на вопрос о компании {company_name}, используя ТОЛЬКО информацию из предоставленных источников.

Вопрос: {user_input}

Источники:
{sources_text}

Дай краткий и точный ответ, указывая конкретные цифры из источников."""
            else:
                # Промпт для других тем
                prompt = f"""Ответь на вопрос, используя ТОЛЬКО информацию из предоставленных источников.

Вопрос: {user_input}

Источники:
{sources_text}

Дай структурированный ответ на основе источников."""

            # Генерация ответа
            response = st.session_state.llm.invoke(prompt)
            with st.chat_message("assistant"):
                st.write(response.content)
            st.session_state.chat_manager.add_message(
                st.session_state.chat_manager.current_chat_id,
                "assistant",
                response.content,
                temperature
            )
            st.stop()

        except Exception as e:
            logger.error(f"Ошибка при поиске или генерации ответа: {str(e)}")
            st.error("Произошла ошибка при обработке запроса")
            st.stop()
