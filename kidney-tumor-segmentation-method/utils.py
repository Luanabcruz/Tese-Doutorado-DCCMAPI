

import os
import numpy as np
from PIL import Image
from datahandler import get_dataloader_kits19_folder
import segmentation_models_pytorch as smp
import lua_enums as l_enum
import json


def load_case_from_image_folder(folder_path, case_code, image_folder='Images', mask_folder='Masks', tensor_shape=True):
    case = {
        'image': [],
        'mask': [],
        'case_name': [],
        'mask_name': []
    }

    # primeiro, pego a lista de todas as fatias do caso informado (case code)
    filename_template = "case_{:05d}-"
    files = os.listdir(os.path.join(folder_path, image_folder))
    case_filenames = []
    for filename in files:
        if filename_template.format(case_code) in filename:
            case_filenames.append(filename)

    # funcao necessária pois quero que as fatias estejam ordenadas

    def func(name):
        num = int(''.join(x for x in name if x.isdigit()))
        return num

    case_filenames = sorted(case_filenames, key=func)

    for case_name in case_filenames:

        im = Image.open(os.path.join(folder_path, image_folder, case_name))
        mask = np.asarray(Image.open(os.path.join(
            folder_path, mask_folder, 'masc_'+case_name)))

        if tensor_shape:
            im = im.convert('RGB')
            im = np.array(im).transpose(2, 0, 1).reshape(
                1, 3, im.width, im.height)

        case['image'].append(im)
        case['mask'].append(mask)
        case['case_name'].append(case_name)
        case['mask_name'].append('masc_'+case_name)

    case['image'] = np.asarray(case['image'])
    case['mask'] = np.asarray(case['mask'], dtype=np.uint8)

    return case


def create_nested_dir(log_path):
    # Create the experiment directory if not present
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'checkpoint'))


def get_criterion_by_name(loss_name):
    import torch

    from losses.dice_loss import diceloss
    from losses.bce_dice_loss import DiceBCELoss
    from losses.focal_loss import FocalLoss
    from losses.iou_loss import IoULoss

    from losses.dices_loss import WBCE_DiceLoss
    from losses.tversky import TverskyLoss

    if loss_name == 'mse':
        return torch.nn.MSELoss(reduction='mean')
    elif loss_name == 'dice':
        return diceloss()
    elif loss_name == 'bce_dice':
        return WBCE_DiceLoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'jaccard':
        return IoULoss()
    elif loss_name == 'tversky':
        return TverskyLoss()
    else:
        raise AssertionError(
            "\nLoss {} não foi implementada!\n".format(loss_name))


def get_segmentation_model(backbone, cnn, in_channels=1):

    model = None
    encoder_weights = 'imagenet'
    # encoder_weights = None

    if encoder_weights is None:
        print("Sem pré treino")
    else:
        print("Pesos "+encoder_weights)

    if cnn == 'unet':

        model = smp.Unet(backbone, classes=1, in_channels=in_channels, activation='sigmoid',
                         encoder_weights=encoder_weights)
                         
    elif cnn == 'deeplabv3+':
        model = smp.DeepLabV3Plus(backbone, classes=1, in_channels=in_channels, activation='sigmoid',
                                  encoder_weights=encoder_weights)
    elif cnn == 'deeplabv3':
        model = smp.DeepLabV3(backbone, classes=1, in_channels=in_channels, activation='sigmoid',
                              encoder_weights=encoder_weights)
    elif cnn == 'pan':
        model = smp.PAN(backbone, classes=1, in_channels=in_channels, activation='sigmoid',
                        encoder_weights=encoder_weights)
    elif cnn == 'psp':
        model = smp.PSPNet(backbone, classes=1, in_channels=in_channels, activation='sigmoid',
                           encoder_weights=encoder_weights)

    return model


def get_data_loaders(data_aug, cases, dataset_dir, batch_size):
    dataloaders = {}

    # Arquivo do balanceamento de fatias. É aplicado apenas no treino
    with open('./configuration_files/balanced_slices.json') as json_file:
        print("# Balanceamento de Fatias ON")
        balanced_filelist = json.load(json_file)
    # print("# Balanceamento de Fatias OFF")
    # balanced_filelist = None
    # Data augmentation é utilizado apenas no treino
    # cases = ['case_00005', 'case_00008', 'case_00056']
    cases = cases['test']
    dataloaders['Train'] = get_dataloader_kits19_folder(
        dataset_dir, data_aug, cases=cases, balanced_filelist=balanced_filelist, batch_size=batch_size)

    dataloaders['Valid'] = get_dataloader_kits19_folder(
        dataset_dir, l_enum.DataAug.NONE, cases=cases, batch_size=batch_size)

    return dataloaders


def get_log_path(data_aug, cnn, loss_name, backbone, dataset_spreed_id, in_channels, sufix=''):

    log_folder = 'luanet_weights'

    if data_aug == l_enum.DataAug.ONLINE:
        log_folder += "_online_aug"
    elif data_aug == l_enum.DataAug.OFFLINE:
        log_folder += "_offline_aug"
    elif data_aug == l_enum.DataAug.NONE:
        log_folder += "_sem_aug"

    return '../{}/{}_{}_{}_dist_{}_ch_{}{}'.format(
        log_folder, cnn, loss_name, backbone, dataset_spreed_id, in_channels, sufix)


if __name__ == '__main__':
    # load_case_from_image_folder('./kits19_lua/Test', 9)

    model = smp.Unet('resnet101', classes=1, in_channels=3, activation='sigmoid',
                     encoder_weights='imagenet').cuda()

    from torchsummary import summary

    summary(model, (3, 256, 256))
