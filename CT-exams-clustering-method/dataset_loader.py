 
from abc import ABC, abstractmethod
 
class DatasetLoader(ABC):
 
    @abstractmethod
    def getCasesName(self):
        pass

    @abstractmethod
    def getCaseDataByIndex(self, index):
        pass

    @abstractmethod
    def getCaseDataByName(self, name):
        pass

    @abstractmethod
    def getAllsCases(self, name):
        pass