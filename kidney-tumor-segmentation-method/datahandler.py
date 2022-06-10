from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import cv2
import torch
from torchvision import transforms, utils
import json

import imageio
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import random
import lua_enums as l_enum


class SegNumpyDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, root_dir, cases, imageFolder, maskFolder, data_aug, cases_number_format=False, transform=None, balanced_filelist=None):
        self.in_channels  = 3
        self.root_dir = root_dir
        self.transform = transform
        self.data_aug = data_aug

        if cases_number_format:
            cases_names = ["case_{:05d}".format(i) for i in cases]
        else:
            cases_names = cases

        image_names = []
        mask_names = []

        if balanced_filelist is None:
            for case in cases_names:
                image_names.extend(glob.glob(os.path.join(
                    self.root_dir, case, imageFolder, '*')))
                mask_names.extend(glob.glob(os.path.join(
                    self.root_dir, case, maskFolder, '*')))
        else:
            # Essa condição é necessária, pois no data aug offline o nome dos arquivos muda.
            if data_aug != l_enum.DataAug.OFFLINE:
                for case in cases_names:
                    image_list = set(os.listdir(os.path.join(
                        self.root_dir, case, imageFolder)))
                    set_balanced = set(balanced_filelist)

                    image_list = list(set_balanced.intersection(image_list))
                    fullpath_image_list = [os.path.join(self.root_dir, case, imageFolder, x)
                                           for x in image_list]
                    fullpath_mask_list = [os.path.join(self.root_dir, case, maskFolder, "masc_"+str(x))
                                          for x in image_list]

                    image_names.extend(fullpath_image_list)
                    mask_names.extend(fullpath_mask_list)
            else:
                for case in cases_names:
                    image_list = set(os.listdir(os.path.join(
                        self.root_dir, case, imageFolder)))

                    balanced_filelist_aug = []
                    # adiciona os data aug manualmente

                    for fl in balanced_filelist:
                        for i in range(0, 5):
                            # case_00000-0-aug-0
                            balanced_filelist_aug.append(
                                "{}-aug-{}.npz".format(fl.replace(".npz", ""), i))

                    set_balanced = set(balanced_filelist_aug)

                    image_list = list(set_balanced.intersection(image_list))
                    fullpath_image_list = [os.path.join(self.root_dir, case, imageFolder, x)
                                           for x in image_list]
                    fullpath_mask_list = [os.path.join(self.root_dir, case, maskFolder, "masc_"+str(x))
                                          for x in image_list]

                    image_names.extend(fullpath_image_list)
                    mask_names.extend(fullpath_mask_list)

        self.image_names = sorted(image_names)
        self.mask_names = sorted(mask_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image = np.load(self.image_names[idx])
        mask = np.load(self.mask_names[idx])

        

        __, file_extension = os.path.splitext(self.image_names[idx])

        if file_extension == '.npz':
            image = image['arr_0']
            mask = mask['arr_0']


        if self.in_channels == 1:
            image = image[1]
        

        # if image.ndim == 2:
        #     image = np.stack((image,)*3, axis=-1)
        #     image = image.transpose(2, 1, 0)

        if self.data_aug == l_enum.DataAug.ONLINE:

            segmap = SegmentationMapsOnImage(mask, shape=(256, 256))

            seq = iaa.Sequential([
                # iaa.CoarseDropout(0.1, size_percent=0.2),  # coarse dropout
                iaa.Affine(
                    scale=(0.5, 1.2),
                    rotate=(-15, 15)
                ),  # rotate the image
                iaa.Flipud(0.5),
                iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                iaa.Sometimes(
                    0.1,
                    iaa.GaussianBlur((0.1, 1.5)),
                ),
                iaa.Sometimes(
                    0.1,
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                )
            ])

            image = image.transpose(1, 2, 0)
            # Apply augmentations for image and mask
            image, mask = seq(image=image, segmentation_maps=segmap)
            image = image.copy()
            mask = mask.copy()
            image = image.transpose(2, 0, 1)
            mask = mask.get_arr()

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, imageresize, maskresize):
        self.imageresize = imageresize
        self.maskresize = maskresize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)
        if len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, self.maskresize, cv2.INTER_AREA)
        image = cv2.resize(image, self.imageresize, cv2.INTER_AREA)
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)

        return {'image': image,
                'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, maskresize=None, imageresize=None):
        image, mask = sample['image'], sample['mask']
        if len(mask.shape) == 2:
            mask = mask.reshape((1,)+mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,)+image.shape)
        return {'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(mask).float()}


class Normalize(object):
    '''Normalize image'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)/255,
                'mask': mask.type(torch.FloatTensor)/255}


def get_dataloader_kits19_folder(data_dir, data_aug, cases=[], balanced_filelist=None, imageFolder='Images', maskFolder='Masks', batch_size=4):

    data_transforms = {
        'Train': transforms.Compose([ToTensor()]),
        'Test': transforms.Compose([ToTensor()]),
    }

    if data_aug == l_enum.DataAug.OFFLINE:
        imageFolder += '_Aug'
        maskFolder += '_Aug'

    image_dataset = SegNumpyDataset(
        data_aug=data_aug, root_dir=data_dir, cases=cases, transform=data_transforms['Train'], maskFolder=maskFolder, imageFolder=imageFolder, balanced_filelist=balanced_filelist)

    dataloader = DataLoader(
        image_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)

    return dataloader
