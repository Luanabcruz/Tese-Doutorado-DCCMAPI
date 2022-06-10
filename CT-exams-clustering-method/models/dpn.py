import torch
from models.deep_feature import DeepFeature
import torchvision.models as models
import torch.nn as nn
import models.torch_utils as tu
import numpy as np
import pretrainedmodels
import pretrainedmodels.utils as utils

class DPN(DeepFeature):

    def __init__(self, model) -> None:
        self.__model = model
        self.__model.to(tu.getDevice())
        self.__model.eval()

    def features(self, data: np) -> torch.Tensor:

        for param in self.__model.parameters():
            param.requires_grad = False

        self.__model.last_linear = utils.Identity()

        input = tu.transform_input(data).float()
        input = input.to(tu.getDevice())

        return self.__model(input).flatten()

    def num_features(self) -> int:
        return len(self.features(np.ones((1,3,512,512))).flatten())

class DPN131(DPN):
    
    def __init__(self) -> None:
        super().__init__(pretrainedmodels.__dict__['dpn131'](pretrained='imagenet')) 