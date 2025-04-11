import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def train_tfidf(corpus_path: str = "corpus.csv"):
    """Обучение TF-IDF векторайзера на корпусе текстов"""
    # Загрузка корпуса
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Файл корпуса {corpus_path} не найден")
    
    df = pd.read_csv(corpus_path)
    corpus = df["chunk_text"].tolist()
    
    # Обучение TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    
    # Сохранение обученного вектора
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("TF-IDF векторайзер успешно обучен и сохранен")

if __name__ == "__main__":
    train_tfidf() 