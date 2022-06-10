from cache_management.cache import Cache
import numpy as np
import os

class NiftCache(Cache):

    def __init__(self, path = './cache/tumor512_vol'):
        super().__init__(path)
    def __get_filepath(self, case_name):
        return os.path.join(self._path, case_name + ".npz")

    def save(self, case_name, vol):
        np.savez(self.__get_filepath(case_name), vol)
    
    def load(self, case_name):
        if self.is_file_exists(case_name):
            return np.load(self.__get_filepath(case_name))['arr_0']
        return None    

    def is_file_exists(self, case_name):
        return os.path.isfile(self.__get_filepath(case_name)) 

