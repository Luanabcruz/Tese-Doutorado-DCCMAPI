import numpy as np
import SimpleITK as sitk
import cv2

def windowing(vol) -> np:
    hu_max = 500
    hu_min = -200
    vol = np.clip(vol, hu_min, hu_max)
    return vol

def apply_mask_on_image_by_label(np_image, np_gt, label) -> np:
    new_image = np.zeros(np_image.shape)
    np.copyto(new_image, np_image, where=(np_gt == label))
    return new_image

def remove_empty_slices_by_label(np_image, np_gt, label):
    ids = []
    for i in range(np_image.shape[2]):
        if(np.amax(np_gt[:, :, i]) != label):
            ids.append(i)
    new_image = np.delete(np_image, ids, 2)
    new_gt = np.delete(np_gt, ids, 2)
    return new_image, new_gt


def rescale_intensity(np_image, rgb_scale=True):
    filter = sitk.RescaleIntensityImageFilter()

    new_image = sitk.GetImageFromArray(np_image)
    if rgb_scale:
        new_image = filter.Execute(new_image, 0, 255)
    else:
        new_image = filter.Execute(new_image, 0, 1)

    new_image = sitk.GetArrayFromImage(new_image)

    return new_image    


def resize_vol(np_image, width=224, height=224):

    np_image = np_image.astype(np.float64)

    nova_altura = height
    nova_largura = width

    if((np_image.shape[1] > height) or (np_image.shape[2] > width)):
        if(np_image.shape[2] > np_image.shape[1]):
            nova_largura = width
            nova_altura = round((nova_largura/np_image.shape[2]) * np_image.shape[1])
        else:
            nova_altura = height
            nova_largura = round((nova_altura/np_image.shape[1]) * np_image.shape[2])

    np_images = np.zeros((len(np_image), nova_altura, nova_largura))

    for i in range(len(np_image)):
        np_images[:, :, i] = cv2.resize(
            np_image[:, :, i], (nova_largura, nova_altura), interpolation=cv2.INTER_LINEAR)

    return np_images