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
from db import (create_tables, save_message, get_chat_messages, archive_chat,
                get_chats, save_chat, update_chat_last_active, update_chat_name,
                archive_messages)
import retrieve
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

    st.divider()
    st.header("Настройки Pinecone DB")
    pinecone_db_name = st.text_input("Имя Pinecone DB", value=os.environ.get("INDEX_NAME", ""))
    pinecone_api_key = st.text_input("Pinecone API ключ", value=os.environ.get("PINECONE_API_KEY", ""), type="password")

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

        st.sidebar.success("API настроен успешно!")
    except Exception as e:
        st.sidebar.error(f"Ошибка настройки API: {str(e)}")

# После получения истории сообщений
current_messages = st.session_state.chat_manager.get_chat_history(st.session_state.chat_manager.current_chat_id)
st.sidebar.write(f"Debug: Found {len(current_messages)} messages in current chat")

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
    try:
        st.session_state.chat_manager.add_message(
            st.session_state.chat_manager.current_chat_id,
            "user",
            user_input,
            temperature
        )
        # Обновляем время последней активности
        update_chat_last_active(st.session_state.chat_manager.current_chat_id)
    except Exception as e:
        logger.error(f"Ошибка при сохранении сообщения пользователя: {e}")
        st.error("Ошибка при сохранении сообщения")

    #Переформулирование запроса для рага. Помогает подтягивать контекст
    q_prompt = "Ты являешься частью системы по ответам на вопросы. Пользователь даст тебе вопрос, который может быть плохо сформулирован. Твоя задача привести его к виду, где 1) Будут отсутствовать все лишние слова (междометия, слова паразиты и прочие) 2) Где будет чёткая формулировка, какую конкретно информацию надо найти (опираясь на историю сообщений в том числе). Если непонятно, что искать, выведи максимально похожий запрос. Если речь идёт о каких-то финансовых отчётностях постарайся упомянуть имя компании.  В ответе должен быть только запрос, без пояснений и обоснований\n" + user_input
    q_mes = HumanMessage(content=q_prompt)
    current_messages.append(q_mes)
    user_input_q = st.session_state.llm.invoke(current_messages).content
    print(user_input_q)
    current_messages.pop()

    #update_chat_last_active(st.session_state.current_chat_id)
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                if "llm" in st.session_state:
                    # Определяем тему запроса пользователя
                    print(user_input_q)
                    topic_result = detect_topic_combined(user_input_q)
                    st.info(f"Определённая тема: {topic_result['topic_name']} "
                           f"(код {topic_result['topic']}, уверенность: {topic_result['confidence']:.2f})\n"
                           f"Причина: {topic_result['reasoning']}")

                    # Получаем текущую тему чата
                    current_chat = st.session_state.chat_manager.get_chat_info(
                        st.session_state.chat_manager.current_chat_id
                    )
                    current_topic = current_chat.get("topic") if current_chat else None

                    # Сохраняем тему, если она определена и отличается от текущей
                    if topic_result["topic"] != 9 and (not current_topic or current_topic != topic_result["topic_name"]):
                        st.session_state.chat_manager.update_topic(
                            st.session_state.chat_manager.current_chat_id,
                            topic_result["topic_name"]
                        )

                    # Используем код темы для RAG-ветки
                    if topic_result["topic"] != 9:
                        # Если тема определена, запускаем RAG-ветку с фильтром по теме
                        
                        # Собираем контекст из предыдущих сообщений
                        context = ""
                        for msg in current_messages[-5:]:  # Берем последние 5 сообщений
                            if isinstance(msg, HumanMessage):
                                context += f"Вопрос: {msg.content}\n"
                            else:
                                context += f"Ответ: {msg.content}\n"
                        
                        try:
                            # Используем TF-IDF только для темы -2, для остальных используем эмбеддинги
                            use_tfidf = topic_result["topic"] == -2
                            
                            logger.info(f"Используем TF-IDF: {use_tfidf}, тема: {topic_result['topic']}, уверенность: {topic_result['confidence']:.2f}")
                            
                            res = retrieve.retrieve(
                                query_text=user_input_q,
                                context=context,
                                topic_code=topic_result["topic"],
                                use_tfidf=use_tfidf
                            )
                        except Exception as e:
                            logger.warning(f"Ошибка при поиске: {str(e)}. Используем эмбеддинги...")
                            # Если возникла ошибка, используем эмбеддинги
                            res = retrieve.retrieve(
                                query_text=user_input_q,
                                topic_code=topic_result["topic"],
                                use_tfidf=False
                            )

                        max_score = max([match["score"] for match in res["matches"]]) if res["matches"] else 0
                        # Используем более низкий порог для TF-IDF
                        SIMILARITY_THRESHOLD = 0.2 if use_tfidf else 0.55
                        
                        logger.info(f"Максимальный score: {max_score:.4f}, порог: {SIMILARITY_THRESHOLD}")

                        if not res["matches"] or max_score < SIMILARITY_THRESHOLD:
                            fallback_msg = "❗️К сожалению, в нашей базе данных нет ответа на этот вопрос. Я обращаюсь к открытым источникам..."
                            st.write(fallback_msg)
                            st.session_state.chat_manager.add_message(
                                st.session_state.chat_manager.current_chat_id,
                                "assistant",
                                fallback_msg,
                                temperature
                            )

                            current_messages = st.session_state.chat_manager.get_chat_history(
                                st.session_state.chat_manager.current_chat_id
                            )
                            response = st.session_state.llm.invoke(current_messages)
                            st.write(response.content)
                            st.session_state.chat_manager.add_message(
                                st.session_state.chat_manager.current_chat_id,
                                "assistant",
                                response.content,
                                temperature
                            )
                            st.stop()
                    else:
                        # Если тема не определена или это общий диалог, используем LLM напрямую
                        current_messages = st.session_state.chat_manager.get_chat_history(
                            st.session_state.chat_manager.current_chat_id
                        )
                        response = st.session_state.llm.invoke(current_messages)
                        st.write(response.content)
                        st.session_state.chat_manager.add_message(
                            st.session_state.chat_manager.current_chat_id,
                            "assistant",
                            response.content,
                            temperature
                        )
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
                    
                    try:
                        first = res["matches"][0]["metadata"]["chunk_text"]
                        second = res["matches"][1]["metadata"]["chunk_text"]
                        third = res["matches"][2]["metadata"]["chunk_text"]
                        fourth = res["matches"][3]["metadata"]["chunk_text"]
                        fifth = res["matches"][4]["metadata"]["chunk_text"]
                        rag_message = f"Ты должен отвечать ТОЛЬКО по данным тебе источникам (не придумывая ничего от себя если такой информации нет говори не знаю) в формате JSON со следующей структурой: content: \"твой ответ на вопрос\", \"sources\": [{first}, {second}, {third}, {fourth}, {fifth}], \"confidence\": число от 0 до 1\n" + f"{user_input} - вопрос пользователя\n {first} - первый источник\n {second} - второй источник\n {third} - третий источник\n {fourth} - четвёртый источник\n {fifth} - пятый источник\n , note: \"примечание об источниках, если необходимо\""

                        human_msg = HumanMessage(content=rag_message)
                        current_messages.append(human_msg)
                        #save_message(st.session_state.current_chat_id, "user", user_input, temperature)
                        structured_llm = st.session_state.llm.with_structured_output(ChatResponse, method="json_mode")
                        response = structured_llm.invoke(current_messages)
                        content = response.content
                        json_match = re.search(r'({.*})', content, re.DOTALL)
                        if json_match:
                            try:
                                json_str = json_match.group(1)
                                structured_data = json.loads(json_str)
                                if "content" not in structured_data:
                                    structured_data["content"] = content
                                if "sources" not in structured_data:
                                    structured_data["sources"] = []
                                if "confidence" not in structured_data:
                                    structured_data["confidence"] = 0.8
                                if "note" not in structured_data:
                                    structured_data["note"] = ""
                                ai_response = AIMessage(content=structured_data["content"])
                                ai_response.structured_output = structured_data
                                st.write(structured_data["content"])
                                if structured_data.get("note"):
                                    with st.expander("Примечание"):
                                        st.write(structured_data["note"])
                                with st.expander("Структурированный ответ"):
                                    st.json(structured_data)
                                
                                st.session_state.chat_manager.add_message(
                                    st.session_state.chat_manager.current_chat_id,
                                    "assistant",
                                    structured_data["content"],
                                    temperature,
                                    structured_data
                                )
                            except json.JSONDecodeError:
                                st.write(content)
                                st.session_state.chat_manager.add_message(
                                    st.session_state.chat_manager.current_chat_id,
                                    "assistant",
                                    content,
                                    temperature
                                )
                        else:
                            st.write(content)
                            st.session_state.chat_manager.add_message(
                                st.session_state.chat_manager.current_chat_id,
                                "assistant",
                                content,
                                temperature
                            )
                    except Exception as e:
                        st.error(f"Ошибка при обработке структурированного ответа: {str(e)}")
                        current_messages = st.session_state.chat_manager.get_chat_history(
                            st.session_state.chat_manager.current_chat_id
                        )
                        response = st.session_state.llm.invoke(current_messages)
                        st.write(response.content)
                        st.session_state.chat_manager.add_message(
                            st.session_state.chat_manager.current_chat_id,
                            "assistant",
                            response.content,
                            temperature
                        )
                else:
                    raise Exception("LLM не настроен")
            except Exception as e:
                st.error(f"Ошибка при получении ответа: {str(e)}")
                error_msg = "Это тестовый ответ, так как произошла ошибка."
                st.write(error_msg)
                st.session_state.chat_manager.add_message(
                    st.session_state.chat_manager.current_chat_id,
                    "assistant",
                    error_msg,
                    temperature
                )
