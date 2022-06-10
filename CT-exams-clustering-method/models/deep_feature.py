from abc import ABC, abstractmethod

class DeepFeature(ABC):

    @abstractmethod
    def features(self, data):
        pass

    @abstractmethod
    def num_features(self):
        pass

