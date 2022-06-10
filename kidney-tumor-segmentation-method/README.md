# LuaNet Segmentation 2D/2.5D

Adaptação do repositório: https://github.com/msminhas93/DeepLabv3FineTuning e uso da Biblioteca Segmentation Models: https://github.com/qubvel/segmentation_models

# Informações do repositório

Segmentação semantica usando PyTorch. Atualmente são suportadas as seguintes redes:

**Models**

- `Unet <https://arxiv.org/abs/1505.04597>`\_\_
- `FPN <http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf>`\_\_
- `Linknet <https://arxiv.org/abs/1707.03718>`\_\_
- `PSPNet <https://arxiv.org/abs/1612.01105>`\_\_
- `DeepLabV3 `\_\_
- `DeepLabV3+ `\_\_

**Backbones**

| Type         | Names                                                                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| VGG          | `'vgg16' 'vgg19'`                                                                                                                      |
| ResNet       | `'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'`                                                                             |
| SE-ResNet    | `'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'`                                                                   |
| ResNeXt      | `'resnext50' 'resnext101'`                                                                                                             |
| SE-ResNeXt   | `'seresnext50' 'seresnext101'`                                                                                                         |
| SENet154     | `'senet154'`                                                                                                                           |
| DenseNet     | `'densenet121' 'densenet169' 'densenet201'`                                                                                            |
| Inception    | `'inceptionv3' 'inceptionresnetv2'`                                                                                                    |
| MobileNet    | `'mobilenet' 'mobilenetv2'`                                                                                                            |
| EfficientNet | `'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4 'efficientnetb5' efficientnetb6' efficientnetb7'` |


