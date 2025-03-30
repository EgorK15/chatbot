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
from db import (create_tables, save_message, get_chat_messages, archive_chat,
                get_chats, save_chat, update_chat_last_active, update_chat_name,
                archive_messages)
import retrieve
from routing import detect_topic  # Импорт функции для определения темы

# Инициализируем таблицы базы данных
create_tables()

# Определение структуры вывода с помощью Pydantic с дополнительным полем note
class ChatResponse(BaseModel):
    answer: str = Field(description="Ответ на вопрос пользователя")
    sources: list = Field(description="Источники информации, использованные для ответа", default_factory=list)
    confidence: float = Field(description="Уровень уверенности в ответе от 0 до 1", ge=0, le=1)
    note: str = Field(description="Примечание об источниках, если необходимо", default="")

# Настройка заголовка приложения
st.title('Llama Chatbot')

# Инициализация структуры для хранения нескольких чатов
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    stored_chats = get_chats()
    if stored_chats:
        # Загружаем сохраненные чаты из БД
        for chat in stored_chats:
            chat_id, name, created_at, last_active, status, model_name_db, temp = chat
            # Если чат архивирован, скрываем его (не загружаем в список)
            if status == 'archived':
                continue
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
                        except Exception:
                            pass
                    msg_objects.append(ai_msg)
            st.session_state.chat_sessions[chat_id] = {
                "name": name,
                "created_at": created_at,
                "status": status,
                "messages": msg_objects
            }
        # Если у нас остались чаты, выбираем первый как текущий
        if st.session_state.chat_sessions:
            st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
        else:
            # Если все чаты были archived, создаем новый
            first_chat_id = str(uuid.uuid4())
            created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.chat_sessions[first_chat_id] = {
                "name": "Новый чат",
                "messages": [],
                "created_at": created_at,
                "status": "active"
            }
            st.session_state.current_chat_id = first_chat_id
            save_chat(first_chat_id, "Новый чат", created_at)
    else:
        # Если нет сохраненных чатов, создаем новый
        first_chat_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.chat_sessions[first_chat_id] = {
            "name": "Новый чат",
            "messages": [],
            "created_at": created_at,
            "status": "active"
        }
        st.session_state.current_chat_id = first_chat_id
        save_chat(first_chat_id, "Новый чат", created_at)

# Функция для создания нового чата
def create_new_chat():
    chat_id = str(uuid.uuid4())
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.chat_sessions[chat_id] = {
        "name": f"Новый чат {len(st.session_state.chat_sessions) + 1}",
        "messages": [],
        "created_at": created_at,
        "status": "active"
    }
    st.session_state.current_chat_id = chat_id
    save_chat(chat_id, st.session_state.chat_sessions[chat_id]["name"], created_at)

# Функция для удаления текущего чата
def delete_current_chat():
    current_chat = st.session_state.current_chat_id
    archive_messages(current_chat)  # Архивируем сообщения
    archive_chat(current_chat)      # Архивируем чат
    del st.session_state.chat_sessions[current_chat]
    if st.session_state.chat_sessions:
        st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
    else:
        create_new_chat()

# Функция для переименования текущего чата
def rename_current_chat(new_name):
    if new_name:
        st.session_state.chat_sessions[st.session_state.current_chat_id]["name"] = new_name
        update_chat_name(st.session_state.current_chat_id, new_name)

# Боковая панель для настроек
with st.sidebar:
    st.header("Чаты")
    chat_options = {chat_id: f"{chat_data['name']} ({chat_data['created_at']})"
                    for chat_id, chat_data in st.session_state.chat_sessions.items()}
    selected_chat = st.selectbox(
        "Выберите чат",
        options=list(chat_options.keys()),
        format_func=lambda x: chat_options[x],
        index=list(chat_options.keys()).index(st.session_state.current_chat_id)
    )
    if selected_chat != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat
        messages = get_chat_messages(selected_chat)
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
                    except Exception:
                        pass
                msg_objects.append(ai_msg)
        st.session_state.chat_sessions[selected_chat]["messages"] = msg_objects
        st.rerun()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Новый чат"):
            create_new_chat()
            st.rerun()
    with col2:
        if st.button("Удалить чат"):
            delete_current_chat()
            st.rerun()
    new_chat_name = st.text_input(
        "Переименовать чат",
        value=st.session_state.chat_sessions[st.session_state.current_chat_id]["name"]
    )
    if new_chat_name != st.session_state.chat_sessions[st.session_state.current_chat_id]["name"]:
        rename_current_chat(new_chat_name)
    st.divider()
    st.header("Настройки API")
    base = os.environ.get("OPENAI_API_BASE", "base_url")
    key = os.environ.get("OPENAI_API_KEY", "api_key")
    model = os.environ.get("MODEL_NAME", "model")
    temp = os.environ.get("TEMPERATURE", 0.7)
    if temp == "":
        temp = 0.7
    api_key = st.text_input("API ключ", value=key, type="password")
    api_base = st.text_input("API URL", value=base)
    model_name = st.text_input("Название модели", value=model)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=float(temp), step=0.1)
    # Чекбокс "Использовать RAG" удалён – выбор определяется автоматически
    st.divider()
    st.header("Настройки Pinecone DB")
    pinecone_db_name = st.text_input("Имя Pinecone DB", value=os.environ.get("PINECONE_INDEX_NAME", ""))
    pinecone_api_key = st.text_input("Pinecone API ключ", value=os.environ.get("PINECONE_API_KEY", ""), type="password")
    if st.button("Очистить текущий чат"):
        # Archive only the messages in the current chat
        archive_messages(st.session_state.current_chat_id)
        st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"] = []
        st.rerun()

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
    try:
        st.session_state.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=api_base
        )
        st.sidebar.success("API настроен успешно!")
    except Exception as e:
        st.sidebar.error(f"Ошибка настройки API: {str(e)}")

# Отображение истории сообщений текущего чата
current_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]

# Отображение истории сообщений текущего чата
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
    user_input_q = st.session_state.llm.invoke([{"role": "system",
                                                 "content": "Ты являешься частью системы по ответам на вопросы. Пользователь даст тебе вопрос, который может быть плохо сформулирован. Твоя задача привести его к виду, где 1) Будут отсутствовать все лишние слова (междометия, слова паразиты и прочие) 2) Где будет чёткая формулировка, какую конкретно информацию надо найти"},
                                                {"role": "user", "content": user_input}]).content
    #human_msg = HumanMessage(content=user_input)
    #current_messages.append(human_msg)
    #save_message(st.session_state.current_chat_id, "user", user_input, temperature)
    update_chat_last_active(st.session_state.current_chat_id)
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                if "llm" in st.session_state:
                    # Определяем тему запроса пользователя
                    topic_result = detect_topic(user_input_q)
                    st.info(f"Определённая тема: {topic_result['topic_name']} "
                            f"(код {topic_result['topic_code']}, уверенность: {topic_result['confidence']:.2f})\n"
                            f"Причина: {topic_result['reasoning']}")

                    # Используем код темы для RAG-ветки
                    if topic_result["topic_code"] != 9:
                        # Если тема определена, запускаем RAG-ветку с фильтром по теме
                        res = retrieve.retrieve(user_input_q, topic_result["topic_code"])

                        max_score = max([match["score"] for match in res["matches"]]) if res["matches"] else 0
                        SIMILARITY_THRESHOLD = 0.55

                        if not res["matches"] or max_score < SIMILARITY_THRESHOLD:
                            human_msg = HumanMessage(content=user_input)
                            current_messages.append(human_msg)
                            fallback_msg = "❗️К сожалению, в нашей базе данных нет ответа на этот вопрос. Я обращаюсь к открытым источникам..."
                            st.write(fallback_msg)
                            ai_notice = AIMessage(content=fallback_msg)
                            current_messages.append(ai_notice)
                            save_message(st.session_state.current_chat_id, "assistant", fallback_msg, temperature)

                            response = st.session_state.llm.invoke(current_messages)
                            st.write(response.content)
                            ai_response = AIMessage(content=response.content)
                            current_messages.append(ai_response)
                            save_message(st.session_state.current_chat_id, "assistant", response.content, temperature)
                            update_chat_last_active(st.session_state.current_chat_id)
                            st.stop()
                    else:
                        # Если тема не определена, ищем по всей базе
                        res = retrieve.retrieve(user_input)

                        max_score = max([match["score"] for match in res["matches"]]) if res["matches"] else 0
                        SIMILARITY_THRESHOLD = 0.55

                        if not res["matches"] or max_score < SIMILARITY_THRESHOLD:
                            fallback_msg = "❗️Не удалось найти информацию в базе. Я обращаюсь к открытым источникам..."
                            st.write(fallback_msg)
                            ai_notice = AIMessage(content=fallback_msg)
                            current_messages.append(ai_notice)
                            save_message(st.session_state.current_chat_id, "assistant", fallback_msg, temperature)

                            response = st.session_state.llm.invoke(current_messages)
                            st.write(response.content)
                            ai_response = AIMessage(content=response.content)
                            current_messages.append(ai_response)
                            save_message(st.session_state.current_chat_id, "assistant", response.content, temperature)
                            update_chat_last_active(st.session_state.current_chat_id)
                            st.stop()

                    # --- RAG ответ ---
                    parser = PydanticOutputParser(pydantic_object=ChatResponse)
                    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
                    system_message = (
                        "Ты помощник, который отвечает на вопрос на основе текстов, которые тебе дадут. Не придумывай ничего, чего бы не было в текстах. "
                        f"Используй следующий формат для ответа:\n{format_instructions}"
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_message),
                        ("user", "{input}")
                    ])
                    st.session_state.debug_info = {
                        "format_instructions": format_instructions,
                        "system_message": system_message
                    }
                    try:

                        res = retrieve.retrieve(user_input_q)
                        first = res["matches"][0]["metadata"]["chunk_text"]
                        second = res["matches"][1]["metadata"]["chunk_text"]
                        third = res["matches"][2]["metadata"]["chunk_text"]
                        fourth = res["matches"][3]["metadata"]["chunk_text"]
                        fifth = res["matches"][4]["metadata"]["chunk_text"]
                        print(user_input_q)
                        rag_message = f"Ты должен отвечать ТОЛЬКО по данным тебе источникам (не придумывая ничего от себя если такой информации нет говори не знаю) в формате JSON со следующей структурой: answer: \"твой ответ на вопрос\", \"sources\": [{first}, {second}, {third}, {fourth}, {fifth}], \"confidence\": число от 0 до 1\n" + f"{user_input} - вопрос пользователя\n {first} - первый источник\n {second} - второй источник\n {third} - третий источник\n {fourth} - четвёртый источник\n {fifth} - пятый источник\n , note: \"примечание об источниках, если необходимо\""
                        human_msg = HumanMessage(content=rag_message)
                        current_messages.append(human_msg)
                        save_message(st.session_state.current_chat_id, "user", rag_message, temperature)
                        response = st.session_state.llm.invoke(current_messages)
                        content = response.content
                        json_match = re.search(r'({.*})', content, re.DOTALL)
                        if json_match:
                            try:
                                json_str = json_match.group(1)
                                structured_data = json.loads(json_str)
                                if "answer" not in structured_data:
                                    structured_data["answer"] = content
                                if "sources" not in structured_data:
                                    structured_data["sources"] = []
                                if "confidence" not in structured_data:
                                    structured_data["confidence"] = 0.8
                                if "note" not in structured_data:
                                    structured_data["note"] = ""
                                ai_response = AIMessage(content=structured_data["answer"])
                                ai_response.structured_output = structured_data
                                st.write(structured_data["answer"])
                                if structured_data.get("note"):
                                    with st.expander("Примечание"):
                                        st.write(structured_data["note"])
                                with st.expander("Структурированный ответ"):
                                    st.json(structured_data)
                                current_messages.pop()
                                true_human_msg = HumanMessage(content=user_input)
                                current_messages.append(true_human_msg)
                                current_messages.append(ai_response)
                                save_message(st.session_state.current_chat_id, "assistant", response.content, temperature, json.dumps(structured_data))
                            except json.JSONDecodeError:
                                st.write(content)
                                ai_response = AIMessage(content=content)
                                current_messages.append(ai_response)
                                save_message(st.session_state.current_chat_id, "assistant", content, temperature)
                        else:
                            st.write(content)
                            ai_response = AIMessage(content=content)
                            current_messages.append(ai_response)
                            save_message(st.session_state.current_chat_id, "assistant", content, temperature)
                    except Exception as e:
                        st.error(f"Ошибка при обработке структурированного ответа: {str(e)}")
                        response = st.session_state.llm.invoke(current_messages)
                        st.write(response.content)
                        ai_response = AIMessage(content=response.content)
                        current_messages.append(ai_response)
                        save_message(st.session_state.current_chat_id, "assistant", response.content, temperature)
                    update_chat_last_active(st.session_state.current_chat_id)
                else:
                    raise Exception("LLM не настроен")
            except Exception as e:
                st.error(f"Ошибка при получении ответа: {str(e)}")
                st.write("Это тестовый ответ, так как произошла ошибка.")
                ai_response = AIMessage(content="Это тестовый ответ, так как произошла ошибка.")
                current_messages.append(ai_response)
                save_message(st.session_state.current_chat_id, "assistant", "Это тестовый ответ, так как произошла ошибка.", temperature)