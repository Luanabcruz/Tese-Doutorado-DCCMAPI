import os
from abc import ABC, abstractmethod

class Cache:

    def __init__(self, path):
        self._path = path
        self.__initCacheDir()

    def __initCacheDir(self):
        try:
            os.makedirs(self._path)
        except:
            pass
    @abstractmethod        
    def save():
        pass 

    @abstractmethod        
    def load():
        pass   

