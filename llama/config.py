import os

#guide me ()

def my_dir(folder_name):
    # Получаем текущий рабочий каталог
    current_dir = os.getcwd()
    # Строим путь к указанной папке, находящейся на три уровня вверх
    target_path = os.path.join(current_dir, '..', '..', folder_name)
    # Нормализуем путь (убираем лишние '..' и т.д.)
    return os.path.normpath(target_path)+"\\"

def path_me (file_name):
    # Получаем текущий рабочий каталог
    current_dir = os.getcwd()
    # Строим путь на три уровня вверх
    config_path = os.path.join(current_dir, '..', '..', file_name)
    # Нормализуем путь (убираем лишние '..' и т.д.)
    return os.path.normpath(config_path)

    
def config_me ():
    # Читаем файл
    with open(path_me('llama.config'), 'r') as file:
        api_base = file.readline().strip()
        my_model = file.readline().strip()
        API_KEY = file.readline().strip()   
        YOUR_API_KEY = file.readline().strip()
        index_name = file.readline().strip()
 
    return API_KEY, api_base, my_model, YOUR_API_KEY, index_name