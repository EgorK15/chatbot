# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from tfidf_constants import TFIDF_CONFIG, RELEVANCE_THRESHOLDS

class TopicClassifier:
    def __init__(self):
        # Корпус документов для каждой темы
        self.topic_docs = {
            -2: [
                "rectifier technologies",
                "rectifier", 
                "brave bison", 
                "starvest",
                "revenue",
                "financial",
                "company",
                "corporation",
                "business",
                "enterprise",
                "firm",
                "organization",
                "revenue 2022",
                "financial report",
                "annual report"
            ],
            
            -1: [
                "высшее образование университет институт",
                "бакалавриат магистратура",
                "диплом вуз учеба",
                "баклавр",
                "магистратура",
                "бакалавриат",
                "вуз",
                "университет",
                "институт"
            ],
            
            0: [
                "французский бульдог собака",
                "бульдог",
                "про бульдогов",
                "щенок бульдог порода",
                "собака питомец французский",
                "пес щенок дог",
                "собаководство кинология дрессировка",
                "выгул собаки прогулка",
                "бульдожка щенки французы",
                "бульдог щенок",
                "бульдог щенок порода",
                "бульдог щенок французский",
                "бульдог щенок французский порода",
                "бульдог щенок французский порода порода",
                "бульдог щенок французский порода порода порода"
            ],
            
            3: [
                "red mad robot компания",
                "rmr рмр разработка",
                "red mad robot digital",
                "red mad robot",
                "рмр"
            ],
            
            4: [
                "telegram мессенджер",
                "телеграм бот канал",
                "чат группа телеграм",
                "telegram messenger",
                "тг телега сообщения",
                "телеграм",
                "телеграм бот",
                "телеграм чат",
                "телеграм группа",
                "телеграм сообщения",
                "тг телега"
            ],
            
            1: [
                "уход за кожей косметика",
                "ретинол spf крем",
                "ретинол",
                "крем маска пилинг",
                "очищение тоник сыворотка",
                "косметология уход",
                "кожа лицо процедуры",
                "spf",
                "кожа лица"
            ],
            
            2: [
                "мостостроительство конструкция",
                "строительство мостов",
                "мост опора пролет",
                "конструкция строительство",
                "СП35.13330.2010"
            ],
            
            5: [
                "michelin рейтинг ресторанов",
                "мишлен звезды еда",
                "рестораны рейтинг",
                "мишлен",
                "michelin"
                
            ],
            
            6: [
                "трудовой кодекс",
                "тк рф",
                "увольнение отпуск больничный",
                "трудовой договор работа",
                "сотрудник работник права"
            ],
            
            7: [
                "world class",
                "фитнес",
                "ворлд класс тренировка",
                "спортзал тренажерный зал",
                "фитнес клуб world class",
                "тренер спорт зал"
            ],
            
            8: [
                "оплата услуг",
                "дом ру",
                "мтс банк",
                "платеж квитанция",
                "оплата интернет телефон",
                "счет услуги платеж"
            ]
        }
        
        # Маппинг тем на их названия
        self.topic_names = {
            -2: "компании",
            -1: "высшее образование",
            0: "французский бульдог",
            3: "red mad robot",
            4: "telegram",
            1: "уход за кожей",
            2: "мостостроительство",
            5: "michelin",
            6: "трудовой кодекс",
            7: "world class",
            8: "оплата услуг"
        }
        
        # Общие фразы для определения общего диалога
        self.general_patterns = [
            r'\b(привет|здравствуй|добрый день|доброе утро|добрый вечер)\b',
            r'\b(как дела|что нового|пока|до свидания)\b',
            r'\b(спасибо|благодарю|помоги|помогите)\b',
            r'^(?:эй|хей|хай|йоу)$',
        ]
        
        # Инициализация TF-IDF векторизатора с общими параметрами
        self.vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
        
        # Подготовка корпуса и векторизация
        self.prepare_corpus()
        
    def prepare_corpus(self):
        # Подготовка корпуса документов
        self.corpus = []
        self.topic_labels = []
        
        for topic_id, docs in self.topic_docs.items():
            self.corpus.extend(docs)
            self.topic_labels.extend([topic_id] * len(docs))
        
        # Векторизация корпуса
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        
    def is_general_query(self, query):
        """Проверка на общие фразы"""
        return any(re.search(pattern, query.lower()) for pattern in self.general_patterns)
    
    def classify(self, query):
        """Классификация запроса"""
        if self.is_general_query(query):
            return {
                "topic": 9,
                "topic_name": "общий диалог",
                "confidence": 1.0,
                "reasoning": "Обнаружена общая фраза или приветствие"
            }
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        # Используем порог из констант
        if best_score < RELEVANCE_THRESHOLDS['topic']:
            return {
                "topic": 9,
                "topic_name": "общий диалог",
                "confidence": float(best_score),
                "reasoning": "Низкая уверенность в определении темы, используем LLM"
            }
        
        predicted_topic = self.topic_labels[best_match_idx]
        return {
            "topic": predicted_topic,
            "topic_name": self.topic_names[predicted_topic],
            "confidence": float(best_score),
            "reasoning": f"Определено через TF-IDF (score: {best_score:.2f})"
        }

# Создаем экземпляр классификатора
classifier = TopicClassifier()

def detect_topic_combined(query):
    """Основная функция определения темы"""
    return classifier.classify(query.lower())