import os

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Инициализация Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])  # Замените на ваш API-ключ

# Подключение к индексу
index_name = os.environ["INDEX_NAME"]
index = pc.Index(index_name)

# Загрузка модели
model = SentenceTransformer('all-MiniLM-L6-v2')

# Векторизация запроса
def retrieve(query_text):
    query_embedding = model.encode(query_text)
    # Поиск в индексе
    results = index.query(
        vector=query_embedding.tolist(),  # Вектор запроса
        top_k=5,  # Количество возвращаемых результатов
        include_metadata=True  # Включить метаданные в результаты
    )
    # Вывод результатов
    return results

if __name__ == '__main__':
    res = retrieve("Бульдог подходит для жизни в частном доме?")
    first = res["matches"][0]["metadata"]["chunk_text"]
    second = res["matches"][1]["metadata"]["chunk_text"]
    third = res["matches"][2]["metadata"]["chunk_text"]
    fourth = res["matches"][3]["metadata"]["chunk_text"]
    fifth = res["matches"][4]["metadata"]["chunk_text"]
    print(f"a - вопрос пользователя\n {first} - первый источник\n {second} - второй источник\n {third} - третий источник\n {fourth} - четвёртый источник\n {fifth} - пятый источник\n")