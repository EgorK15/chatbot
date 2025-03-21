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

# Определение структуры вывода с помощью Pydantic
class ChatResponse(BaseModel):
    answer: str = Field(description="Ответ на вопрос пользователя")
    sources: list = Field(description="Источники информации, использованные для ответа", default_factory=list)
    confidence: float = Field(description="Уровень уверенности в ответе от 0 до 1", ge=0, le=1)

# Настройка заголовка приложения
st.title('Llama Chatbot')

# Инициализация структуры для хранения нескольких чатов
if "chat_sessions" not in st.session_state:
    # Создаем словарь для хранения всех чатов
    st.session_state.chat_sessions = {}
    
    # Создаем первый чат
    first_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[first_chat_id] = {
        "name": "Новый чат",
        "messages": [],
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    
    # Устанавливаем первый чат как текущий
    st.session_state.current_chat_id = first_chat_id

# Функция для создания нового чата
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = {
        "name": f"Новый чат {len(st.session_state.chat_sessions) + 1}",
        "messages": [],
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    st.session_state.current_chat_id = chat_id
    
# Функция для удаления текущего чата
def delete_current_chat():
    if len(st.session_state.chat_sessions) > 1:
        # Удаляем текущий чат
        del st.session_state.chat_sessions[st.session_state.current_chat_id]
        
        # Выбираем последний чат как текущий
        st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
    else:
        # Если это последний чат, то очищаем его сообщения
        st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"] = []

# Функция для переименования текущего чата
def rename_current_chat(new_name):
    if new_name:
        st.session_state.chat_sessions[st.session_state.current_chat_id]["name"] = new_name

# Создаем боковую панель для настроек
with st.sidebar:
    # Секция управления чатами
    st.header("Чаты")
    
    # Выбор чата
    chat_options = {chat_id: f"{chat_data['name']} ({chat_data['created_at']})" 
                    for chat_id, chat_data in st.session_state.chat_sessions.items()}
    
    selected_chat = st.selectbox(
        "Выберите чат",
        options=list(chat_options.keys()),
        format_func=lambda x: chat_options[x],
        index=list(chat_options.keys()).index(st.session_state.current_chat_id)
    )
    
    # Устанавливаем выбранный чат как текущий
    if selected_chat != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat
        st.rerun()
    
    # Кнопки управления чатами
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Новый чат"):
            create_new_chat()
            st.rerun()
    with col2:
        if st.button("Удалить чат"):
            delete_current_chat()
            st.rerun()
    
    # Переименование чата
    new_chat_name = st.text_input(
        "Переименовать чат", 
        value=st.session_state.chat_sessions[st.session_state.current_chat_id]["name"]
    )
    if new_chat_name != st.session_state.chat_sessions[st.session_state.current_chat_id]["name"]:
        rename_current_chat(new_chat_name)
    
    # Разделитель
    st.divider()
    
    # Настройки API
    st.header("Настройки API")
    api_key = st.text_input("API ключ", value="F9yei26LMobzDdVmdKuEcbIlTod3OoTJXAszIUoMdAwYwGWS50bvKODj99JZRqcJA8mUQgR1zlXoHpCORP97ODLAyfagqrob", type="password")
    api_base = st.text_input("API URL", value="https://llama3gpu.neuraldeep.tech/v1")
    model_name = st.text_input("Название модели", value="llama-3-8b-instruct-8k")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    
    # Опция для структурированного вывода
    use_structured_output = st.checkbox("Использовать структурированный вывод", value=False)
    
    # Кнопка для сброса текущего чата
    if st.button("Очистить текущий чат"):
        st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"] = []
        st.rerun()

# Конфигурация OpenAI-совместимого клиента для Llama
# Проверяем, изменились ли настройки API
api_changed = (
    "llm" not in st.session_state or
    st.session_state.get("api_key", "") != api_key or
    st.session_state.get("api_base", "") != api_base or
    st.session_state.get("model_name", "") != model_name or
    st.session_state.get("temperature", 0) != temperature
)

if api_changed:
    # Сохраняем текущие настройки в session_state
    st.session_state.api_key = api_key
    st.session_state.api_base = api_base
    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    
    # Настраиваем переменные окружения
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    
    # Создаем новый клиент LLM
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

# Получаем сообщения текущего чата
current_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]

# Отображение истории сообщений текущего чата
for message in current_messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            # Проверяем, структурированный ли это ответ
            if hasattr(message, 'structured_output') and message.structured_output:
                st.write(message.content)
                with st.expander("Структурированный ответ"):
                    st.json(message.structured_output)
            else:
                st.write(message.content)

# Поле ввода для нового сообщения
user_input = st.chat_input("Введите ваше сообщение...")

# Обработка ввода пользователя
if user_input:
    # Добавляем сообщение пользователя в историю текущего чата
    current_messages.append(HumanMessage(content=user_input))
    
    # Отображаем сообщение пользователя
    with st.chat_message("user"):
        st.write(user_input)
    
    # Получаем ответ от модели
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                if "llm" in st.session_state:
                    if use_structured_output:
                        # Создаем парсер для структурированного вывода
                        parser = PydanticOutputParser(pydantic_object=ChatResponse)
                        
                        # Получаем инструкции и экранируем фигурные скобки
                        format_instructions = parser.get_format_instructions()
                        # Заменяем одинарные фигурные скобки на двойные для экранирования
                        format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
                        
                        # Создаем системное сообщение с экранированными инструкциями
                        system_message = (
                            "Ты помощник, который отвечает в структурированном формате. "
                            f"Используй следующий формат для ответа:\n{format_instructions}"
                        )
                        
                        # Создаем шаблон для структурированного вывода
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_message),
                            ("user", "{input}")
                        ])
                        
                        # Вывод отладочной информации
                        st.session_state.debug_info = {
                            "format_instructions": format_instructions,
                            "system_message": system_message
                        }
                        
                        # Специальная обработка для простого примера структурированного вывода
                        try:
                            # Вариант с ручным форматированием запроса
                            response = st.session_state.llm.invoke([
                                {"role": "system", "content": "Ты должен отвечать ТОЛЬКО в формате JSON со следующей структурой: {\"answer\": \"твой ответ на вопрос\", \"sources\": [\"источник1\", \"источник2\"], \"confidence\": число от 0 до 1}"},
                                {"role": "user", "content": user_input}
                            ])
                            
                            # Пытаемся извлечь JSON из ответа
                            content = response.content
                            # Находим JSON в ответе с помощью регулярного выражения
                            json_match = re.search(r'({.*})', content, re.DOTALL)
                            
                            if json_match:
                                try:
                                    json_str = json_match.group(1)
                                    structured_data = json.loads(json_str)
                                    
                                    # Проверяем наличие необходимых полей
                                    if "answer" not in structured_data:
                                        structured_data["answer"] = content
                                    if "sources" not in structured_data:
                                        structured_data["sources"] = []
                                    if "confidence" not in structured_data:
                                        structured_data["confidence"] = 0.8
                                    
                                    # Создаем AI сообщение с обычным текстом и структурированными данными
                                    ai_response = AIMessage(content=structured_data["answer"])
                                    ai_response.structured_output = structured_data
                                    
                                    # Отображаем ответ
                                    st.write(structured_data["answer"])
                                    with st.expander("Структурированный ответ"):
                                        st.json(structured_data)
                                        
                                    # Добавляем ответ в историю сообщений текущего чата
                                    current_messages.append(ai_response)
                                    
                                except json.JSONDecodeError:
                                    # Если не удалось распарсить JSON, используем обычный ответ
                                    st.write(content)
                                    current_messages.append(AIMessage(content=content))
                            else:
                                # Если не нашли JSON, используем обычный ответ
                                st.write(content)
                                current_messages.append(AIMessage(content=content))
                                
                        except Exception as e:
                            st.error(f"Ошибка при обработке структурированного ответа: {str(e)}")
                            response = st.session_state.llm.invoke(current_messages)
                            st.write(response.content)
                            current_messages.append(response)
                    else:
                        # Обычный неструктурированный ответ
                        response = st.session_state.llm.invoke(current_messages)
                        st.write(response.content)
                        # Добавляем ответ бота в историю текущего чата
                        current_messages.append(response)
                else:
                    raise Exception("LLM не настроен")
            except Exception as e:
                st.error(f"Ошибка при получении ответа: {str(e)}")
                # Для отладки используйте заглушку
                st.write("Это тестовый ответ, так как произошла ошибка.")
                current_messages.append(AIMessage(content="Это тестовый ответ, так как произошла ошибка.")) 