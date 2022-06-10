import glob
import numpy as np
import cv2
import SimpleITK as sitk
from sklearn.preprocessing import MinMaxScaler
import subprocess
import os


def histogram_matching(volume, template):
    filtro = sitk.MinimumMaximumImageFilter()
    filtro.Execute(template)
    maximo = int(filtro.GetMaximum())
    maximo = 1088  # do case 7

    matcher = sitk.HistogramMatchingImageFilter()
    # matcher.SetNumberOfHistogramLevels(maximo)
    # matcher.SetNumberOfMatchPoints(1)
    # matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(volume, template, maximo, 7, True)
    return moving


def dilation_filter(gt, kernel_value):
    dilation_f = sitk.BinaryDilateImageFilter()
    dilation_f.SetKernelRadius(kernel_value)
    dilation_f.SetForegroundValue(1)
    dilated = dilation_f.Execute(gt)
    dilated = sitk.GetArrayFromImage(dilated)

    return dilated


def curvature_flow_filter(image):
    curvature_flow = sitk.CurvatureFlowImageFilter()
    # curvature_flow.SetNumberOfIterations(5)
    # curvature_flow.SetTimeStep()
    curvature = curvature_flow.Execute(image)
    curvature = sitk.GetArrayFromImage(curvature)

    return curvature


def clahe(imagem_array):
    for i in range(imagem_array.shape[0]):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imagem_array[i, :, :] = clahe.apply(imagem_array[i, :, :])

    return imagem_array


def resize_image(image, width=256, height=256):
    imagens = np.zeros((len(image), width, height))

    for i in range(len(image)):
        imagens[i, :, :] = cv2.resize(
            image[i, :, :], (width, height), interpolation=cv2.INTER_CUBIC)
    return imagens


def remove_empty_slices_sem_rins(images, masks):
    ids = []
    for i in range(images.shape[0]):
        if(np.amax(masks[i]) == 0):
            ids.append(i)
    new_gt = np.delete(masks, ids, 0)
    new_images = np.delete(images, ids, 0)

    return new_images, new_gt


def remove_empty_slices_sem_lesoes(images, masks):
    ids = []
    for i in range(images.shape[0]):
        if(np.amax(masks[i]) != 2):
            ids.append(i)
    new_gt = np.delete(masks, ids, 0)
    new_images = np.delete(images, ids, 0)

    return new_images, new_gt


def remove_empty_slices_predicts(images, masks, masks_pred):
    ids = []
    for i in range(images.shape[0]):
        if(np.amax(masks[i]) == 0 and np.amax(masks_pred) == 0):
            ids.append(i)
    new_gt = np.delete(masks, ids, 0)
    new_images = np.delete(images, ids, 0)

    return new_images, new_gt


def manter_rins_com_lesoes(imagem_array, gt_array):

    imagem_rins = np.zeros(imagem_array.shape)
    np.copyto(imagem_rins, imagem_array, where=(gt_array == 2))

    return imagem_rins


def windowing(imagem_especificada_array):
    array_novo = np.where(imagem_especificada_array >= -
                          200, imagem_especificada_array, -200)
    y_train = np.where(array_novo <= 500, array_novo, 500)

    return y_train


def rescale_intensity(imagem_especificada_array, rgb_scale=True):

    filtro = sitk.RescaleIntensityImageFilter()
    nova_imagem = sitk.GetImageFromArray(imagem_especificada_array)
    # nova_imagem.CopyInformation(imagem)

    if rgb_scale:
        nova_imagem = filtro.Execute(nova_imagem)
    else:
        nova_imagem = filtro.Execute(nova_imagem)

    nova_imagem = sitk.GetArrayFromImage(nova_imagem)

    return nova_imagem


def resize_proporcional_image(image, width=256, height=256):

    nova_altura = height
    nova_largura = width

    if((image.shape[1] > 256) or (image.shape[2] > 256)):
        if(image.shape[2] > image.shape[1]):
            nova_largura = width
            nova_altura = round((nova_largura/image.shape[2]) * image.shape[1])
        else:
            nova_altura = height
            nova_largura = round((nova_altura/image.shape[1]) * image.shape[2])

    imagens = np.zeros((len(image), nova_altura, nova_largura))

    for i in range(len(image)):
        imagens[i, :, :] = cv2.resize(
            image[i, :, :], (nova_largura, nova_altura), interpolation=cv2.INTER_LINEAR)

    return imagens
