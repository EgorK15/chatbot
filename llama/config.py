import os

def config_me ():
    # Получаем текущий рабочий каталог
    current_dir = os.getcwd()

    # Строим путь на три уровня вверх
    config_path = os.path.join(current_dir, '..', '..', 'llama.config')

    # Нормализуем путь (убираем лишние '..' и т.д.)
    config_path = os.path.normpath(config_path)

    # Читаем файл
    with open(config_path, 'r') as file:
        api_base = file.readline().strip()
        my_model = file.readline().strip()
        API_KEY = file.readline().strip()
    
    return API_KEY, api_base, my_model