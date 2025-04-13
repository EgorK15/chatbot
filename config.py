
# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv
from pathlib import Path

# Загружаем переменные из .env файла
env_path = Path('.env')
load_dotenv(dotenv_path=env_path)

# OpenAI конфигурация
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Pinecone конфигурация
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = os.getenv("INDEX_NAME", "red-llama-bertopic")
TFIDF_INDEX_NAME = os.getenv("TFIDF_INDEX_NAME", "red-llama-tf-idf")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "tfidf_data")
