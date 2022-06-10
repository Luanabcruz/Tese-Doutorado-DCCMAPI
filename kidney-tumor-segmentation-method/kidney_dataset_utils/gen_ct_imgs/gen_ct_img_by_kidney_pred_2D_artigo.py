'''
Script para gerar as imagens das lesões com base na marcação do rim pelo especialista
'''


import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import cv2
import glob
import os

from methods.filters import *
from methods.delimitar_regioes_rins import *
from methods.center_images import *

from utils import create_nested_dir

import sys



def generate_case_filename_path(dataset_type, case_name, index, output_dir, is_mask=False, lua_pattern=True):

    file_template = ''

    if lua_pattern:
        if is_mask:
            path = os.path.join(output_dir, 'masc_{}'.format(dataset_type))
            file_template = 'masc_{}-{}'.format(case_name, index)
        else:
            path = os.path.join(output_dir, 'img_{}'.format(dataset_type))
            file_template = '{}-{}'.format(case_name, index)
        # cria diretório caso não exista
        create_nested_dir(path)

        return os.path.join(path, file_template)

    else:

        if is_mask:
            path = os.path.join(output_dir, dataset_type, 'Masks')
            file_template = 'masc_{}-{}'.format(case_name, index)
        else:
            path = os.path.join(output_dir, dataset_type, 'Images')
            file_template = '{}-{}'.format(case_name, index)
        # cria diretório caso não exista
        create_nested_dir(path)

        return os.path.join(path, file_template)


def generate_imgs_masks_2d(path_root, output_dir, local, width, height, only_injuries_slices=False, histogram_specification=False):

    path_root = os.path.join(path_root, local)

    cont = 0
    folders = glob.glob(os.path.join(path_root, '*'))
   
   
    for folder in folders:

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        if nome_paciente  != 'case_00097':
            continue

        folder += "\\"

        print(nome_paciente)
        print("paciente: ", cont)
        cont += 1

        imagem = sitk.ReadImage(folder+"imaging.nii.gz")
        imagem_array = sitk.GetArrayFromImage(imagem)

        gt = sitk.ReadImage(folder+"segmentation.nii.gz")
        gt_array = sitk.GetArrayFromImage(gt)

        imagem_array = imagem_array.transpose(2, 0, 1)
        gt_array = gt_array.transpose(2, 0, 1)

        # especificação do histograma
        if histogram_specification:
            imagem_para_especificar = sitk.GetImageFromArray(imagem_array)
            template = sitk.ReadImage(SPECIFICATION_TEMPLATE)
            imagem_especificada = histogram_matching(
                imagem_para_especificar, template)
            imagem_especificada_array = sitk.GetArrayFromImage(
                imagem_especificada)
        # fim da especificação do histograma

        if histogram_specification:
            imagem_especificada_array = windowing(imagem_especificada_array)
        else:
            imagem_especificada_array = windowing(imagem_array)

        imagem_especificada_array = rescale_intensity(
            imagem_especificada_array, rgb_scale=True)

        # região cortada do rim
        imagem_especificada_array, gt_array = delimitar_rins_imagem_grande(
            imagem_especificada_array, gt_array)
        imagem_especificada_array, gt_array = delimitar_rins_com_textura(
            imagem_especificada_array, gt_array)

        # redimensiona proporcionalmente
        if(gt_array.shape[1] > width or gt_array.shape[2] > height):
            print("ENTROU RESIZE!")
            print(gt_array.shape)
            imagem_especificada_array = resize_proporcional_image(
                imagem_especificada_array, width, height)
            gt_array = resize_proporcional_image(gt_array, width, height)

        if only_injuries_slices:
            # retira fatias que não possuem lesão
            imagem_especificada_array, gt_array = remove_empty_slices_sem_lesoes(
                imagem_especificada_array, gt_array)
        else:
            # para retirar as fatias que não tem rins
            imagem_especificada_array, gt_array = remove_empty_slices_sem_rins(
                imagem_especificada_array, gt_array)

        # centraliza as imagens
        imagem_especificada_array, gt_array = centralizar_imagens(
            imagem_especificada_array, gt_array, width=width, height=height)

        # manter só os rins com lesões
        imagem_especificada_array = manter_rins_com_lesoes(
            imagem_especificada_array, gt_array.astype(np.uint8))

   
        np.save(os.path.join(output_dir,nome_paciente+".npy"), imagem_especificada_array)


if __name__ == '__main__':
    SPECIFICATION_TEMPLATE = "kidney_dataset_utils/imaging_176.nii.gz"

    width = 256
    height = 256

    dataset_type = ""

    dataset_root_dir = r"E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"
 
    output_dir = r'C:\Users\domin\Downloads'

    histogram_specification = False

    generate_imgs_masks_2d(dataset_root_dir, output_dir, dataset_type, width, height,
                           only_injuries_slices=True, histogram_specification=histogram_specification)
