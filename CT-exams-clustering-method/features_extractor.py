 
from abc import ABC, abstractmethod
 
class FeaturesExtractor(ABC):
    def __init__(self, model):
        self._extractor =  model

    @abstractmethod
    def extract(self, data):
        pass
 