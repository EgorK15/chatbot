# -*- coding: utf-8 -*-
from reflection_agent_lang_graph import get_reflective_answer
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
from websearchagent import search_with_agent
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
from gmail_langgraph import search_emails_for_info
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализируем таблицы базы данных
create_tables()

class ChatResponse(BaseModel):
    content: str = Field(description="Ответ на вопрос пользователя")
    sources: list = Field(description="Источники информации, использованные для ответа", default_factory=list)
    confidence: float = Field(description="Уровень уверенности в ответе от 0 до 1", ge=0, le=1)
    note: str = Field(description="Примечание об источниках, если необходимо", default="")

st.title('Llama Chatbot')

if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()

with st.sidebar:
    st.header("Чаты")
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

    col1, col2, col3 = st.columns([0.9, 1, 1.1])
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

user_input = st.chat_input("Введите ваше сообщение...")
if user_input:
    topic_docs = False
    greeting_patterns = [
        r'^\s*(привет|здравствуй|добрый день|доброе утро|добрый вечер|хай|хей|йоу)\s*$',
        r'^\s*(hi|hello|hey|good morning|good afternoon|good evening)\s*$'
    ]
    email_dict = search_emails_for_info(user_input)
    web_dict = search_with_agent(user_input)
    web_dict_use = []
    web_refs = []
    for i in web_dict:
        if i[2] > 0.2:
            web_dict_use.append(i[0])
            web_refs.append(i[1])
    print(web_dict_use)
    if any(re.match(pattern, user_input.lower()) for pattern in greeting_patterns):
        st.session_state.chat_manager.add_message(
            st.session_state.chat_manager.current_chat_id, "user", user_input, temperature
        )
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            greeting_response = "Здравствуйте! Чем могу помочь?"
            st.write(greeting_response)
            st.session_state.chat_manager.add_message(
                st.session_state.chat_manager.current_chat_id, "assistant", greeting_response, temperature
            )
        st.stop()

    with st.chat_message("user"):
        st.write(user_input)
        st.session_state.chat_manager.add_message(
            st.session_state.chat_manager.current_chat_id, "user", user_input, temperature
        )

    with st.spinner("Определяю тему..."):
        topic_result = detect_topic_combined(user_input)
        logger.info(f"Определённая тема: {topic_result['topic_name']} (код {topic_result['topic']}, уверенность: {topic_result['confidence']:.2f})\nПричина: {topic_result['reasoning']}")

    prev_topic_result = None
    for msg in reversed(current_messages):
        if isinstance(msg, HumanMessage):
            prev_topic_result = detect_topic_combined(msg.content)
            break

    if (
        topic_result["topic"] == 9 and
        prev_topic_result and
        prev_topic_result["topic"] in range(-2, 9) and
        topic_result["confidence"] < 0.75
    ):
        logger.info("Тема не определена — наследуем из предыдущего сообщения")
        topic_result = prev_topic_result

    try:
        use_tfidf = topic_result["topic"] == -2
        if use_tfidf:
            translation_response = st.session_state.llm.invoke(f"Переведи на английский: {user_input}")
            final_query = translation_response.content
            logger.info(f"Переведенный запрос: {final_query}")
        else:
            final_query = user_input

        query_text = f"{final_query}"
        logger.info(f"Поисковый запрос: {query_text}")

        info_box = st.empty()
        info_box.info("🔍 Ищу информацию...")

        res = retrieve_docs(query_text=query_text, context="", topic_code=topic_result["topic"])

        info_box.empty()
        src = ""
        for match in res["matches"]:
            logger.info(f"[RAG MATCH] Score: {match['score']:.4f}")
            text = match["metadata"].get("chunk_text", "")
            logger.info(f"[RAG TEXT] ({len(text)} chars): {repr(text[:200])}")
            logger.info(f"[RAG METADATA] topic_name: {match['metadata']['topic_name']}, source: {match['metadata'].get('source_type', 'unknown')}")
            src = match["metadata"].get("source", "")
        if not is_relevant(res, use_tfidf):
            logger.info("Релевантные документы не найдены, генерирую ответ через LLM")
            add = " "
            if email_dict["answer_bool"]:
                add = f"\n\nТакже учти, что {email_dict['answer']}"
            if web_dict_use:
                web_str = " ".join(web_dict_use)
                add += f"\n\nТакже есть информация из интернета, что {web_str}"

            #response = st.session_state.llm.invoke(user_input + add)
            response_str = get_reflective_answer(user_input+add)
            with st.chat_message("assistant"):
                st.write(response_str)
            st.session_state.chat_manager.add_message(
                st.session_state.chat_manager.current_chat_id, "assistant", response_str, temperature
            )
            print(email_dict)
            gmail_answer = ""
            if email_dict["answer_bool"]:
                answer = email_dict["answer"]
                sources = " ".join(email_dict["messages"])
                gmail_answer += "Упоминания найдены в почтовом ящике. В сообщениях (отправитель - тема - id):\n"
                k = 1
                for i in email_dict["messages"]:
                    gmail_answer += f"\n{k}) {i}"
                    k += 1

            if topic_docs:
                gmail_answer += f"\n\nИнформация найдена во внутренних документах, а именно в {src}"

            if web_dict_use:
                for (i,v) in enumerate(web_refs):
                    gmail_answer += f"\n\n{i}) Информация найдена в интернете, а именно в {v}"
            if gmail_answer != "":
                st.info(gmail_answer)
                st.session_state.chat_manager.add_message(st.session_state.chat_manager.current_chat_id,
                                                          "assistant",
                                                          gmail_answer,
                                                          temperature)


            st.stop()
        topic_docs = True
        sources = [match["metadata"]["chunk_text"] for match in res["matches"][:40]]
        sources_text = "\n\n".join(sources)

        if topic_result["topic"] == -2:
            company_name = res["matches"][0]["metadata"].get("topic_name", "")
            prompt = f"""Ты финансовый аналитик. Ответь на вопрос о компании {company_name}, используя ТОЛЬКО информацию из предоставленных источников.
                    
                    
            Вопрос: {user_input}
            
            Источники:
            {sources_text}
            
            Дай краткий и точный ответ, указывая конкретные цифры из источников."""
        else:
            prompt = f"""Ответь на вопрос, используя ТОЛЬКО информацию из архива документов (RAG).

            Вопрос: {user_input}
            
            Контекст:
            {sources_text}
            
            используя ТОЛЬКО информацию из архива документов (RAG)."""
        if email_dict["answer_bool"]:
            prompt += f"\n\nТакже учти, что {email_dict['answer']}"
        #response = st.session_state.llm.invoke(prompt)
        response_str = get_reflective_answer(prompt)
        with st.chat_message("assistant"):
            st.write(response_str)
        st.session_state.chat_manager.add_message(
            st.session_state.chat_manager.current_chat_id, "assistant", response_str, temperature
        )

        print(email_dict)
        gmail_answer = ""
        if email_dict["answer_bool"]:
            answer = email_dict["answer"]
            sources = " ".join(email_dict["messages"])
            gmail_answer += "Упоминания найдены в почтовом ящике. В сообщениях (отправитель - тема - id):\n"
            k = 1
            for i in email_dict["messages"]:
                gmail_answer += f"\n{k}) {i}"
                k+=1

        if topic_docs:
            gmail_answer += f"\n\nИнформация найдена во внутренних документах, а именно в {src}"

        if gmail_answer!="":
            st.info(gmail_answer)
            st.session_state.chat_manager.add_message(st.session_state.chat_manager.current_chat_id,
                                                      "assistant",
                                                      gmail_answer,
                                                      temperature)
        st.stop()

    except Exception as e:
        logger.error(f"Ошибка при поиске или генерации ответа: {str(e)}")
        st.error("Произошла ошибка при обработке запроса")
        st.stop()

    st.stop()