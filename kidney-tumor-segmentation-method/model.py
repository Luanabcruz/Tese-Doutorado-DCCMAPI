""" DeepLabv3 Model download and change the head for your prediction"""
from models.segmentation.deeplabv3 import DeepLabHead

from models.segmentation.fcn import FCNHead

from torchvision import models
from torch import nn


def createDeepLabv3(outputchannels=1, backbone='resnet101', inputchannels=3):

    print("")
    print("DeepLabv3-", backbone)
    print("")

    if backbone == 'resnet101':
        model = models.segmentation.deeplabv3_resnet101(
            pretrained=True, progress=True)

        if inputchannels != 3:
            model.backbone.conv1 = nn.Conv2d(
                inputchannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # x = torch.randn(2, 1, 224, 224)
        # output = model(x)
    elif backbone == 'resnet50':
        model = models.segmentation.deeplabv3_resnet50(
            pretrained=True, progress=True)
    else:
        return NotImplementedError
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model
