import torch
from models.deep_feature import DeepFeature
import torchvision.models as models
import torch.nn as nn
import models.torch_utils as tu
import numpy as np

class Resnet(DeepFeature):

    def __init__(self, model) -> None:
        self.__resnet_model = model
        self.__resnet_model.to(tu.getDevice())

    def features(self, data: np) -> torch.Tensor:

        for param in self.__resnet_model.parameters():
            param.requires_grad = False

        self.__resnet_model.fc = nn.Flatten()

        input = tu.transform_input(data).float()
        input = input.to(tu.getDevice())

        return self.__resnet_model(input)

    def num_features(self) -> int:
        return len(self.features(np.ones((1,3,512,512))).flatten())

class Resnet18(Resnet):
    
    def __init__(self) -> None:
        super().__init__(models.resnet18(pretrained=True)) 

class Resnet34(Resnet):

    def __init__(self) -> None:
        super().__init__(models.resnet34(pretrained=True))

class Resnet50(Resnet):

    def __init__(self) -> None:
        super().__init__(models.resnet50(pretrained=True))     

class Resnet101(Resnet):

    def __init__(self) -> None:
        super().__init__(models.resnet101(pretrained=True))

    
