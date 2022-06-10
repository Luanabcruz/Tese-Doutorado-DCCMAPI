import torch
from models.deep_feature import DeepFeature
import torchvision.models as models
import torch.nn as nn
import models.torch_utils as tu
import numpy as np

class Vgg(DeepFeature):

    def __init__(self, model) -> None:
        self.__vgg_model = model
        self.__vgg_model.to(tu.getDevice())

    def features(self, data: np) -> torch.Tensor:

        for param in self.__vgg_model.parameters():
            param.requires_grad = False

        self.__vgg_model.fc = nn.Flatten()

        input = tu.transform_input(data).float()
        input = input.to(tu.getDevice())

        return self.__vgg_model(input)

    def num_features(self) -> int:
        return len(self.features(np.ones((1,3,512,512))).flatten())

class Vgg11(Vgg):
    
    def __init__(self) -> None:
        super().__init__(models.vgg11(pretrained=True)) 

class Vgg16(Vgg):
    
    def __init__(self) -> None:
        super().__init__(models.vgg16(pretrained=True))  

class Vgg19(Vgg):
    
    def __init__(self) -> None:
        super().__init__(models.vgg19(pretrained=True))  




