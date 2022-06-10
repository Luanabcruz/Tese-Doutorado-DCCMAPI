import os
def create_nested_dir(path, destroy_dir = True):
    # Verifica a existência do diretório, para criá-lo caso não exista.        
    if not os.path.exists(path):
        os.makedirs(path)