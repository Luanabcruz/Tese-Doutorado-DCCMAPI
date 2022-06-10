from cache_management.cache import Cache
import os
import json

class GeneralCache(Cache):

    def __init__(self, path = './cache/general'):
        super().__init__(path)
    def __get_filepath(self, case_name):
        return os.path.join(self._path, case_name + ".json")

    def save(self, keyname, info):        
        with open(self.__get_filepath(keyname), 'w') as f:
            json.dump(info, f)
    
    def load(self, keyname):
        if self.is_file_exists(keyname):
            with open(self.__get_filepath(keyname)) as json_file:
                return  json.load(json_file)

        return None    

    def is_file_exists(self, case_name):
        return os.path.isfile(self.__get_filepath(case_name)) 

