'''
Script para gerar as imagens dos rins com base na marcação do rim pelo especialista
'''


import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import cv2
import glob

from methods.filters import *
from methods.delimitar_regioes_rins import *
from methods.center_images import *

from utils import create_nested_dir


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
    test_cases = ["case_00009", "case_00011", "case_00012", "case_00018", "case_00024", "case_00025", "case_00028", "case_00041", "case_00043", "case_00045", "case_00056", "case_00066", "case_00076", "case_00078", "case_00082",
                  "case_00085", "case_00099", "case_00100", "case_00102", "case_00122", "case_00131", "case_00139", "case_00149", "case_00158", "case_00170", "case_00174", "case_00175", "case_00176", "case_00197", "case_00199", "case_00200"]
    for folder in folders:

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        # if nome_paciente != "case_00165":
        #     continue

        if not nome_paciente in test_cases:
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
            imagem_especificada_array, rgb_scale=False)

        # if False:
        #     # retira fatias que não possuem lesão
        #     imagem_especificada_array, gt_array = remove_empty_slices_sem_lesoes(
        #         imagem_especificada_array, gt_array)
        # else:
        #     # para retirar as fatias que não tem rins
        #     imagem_especificada_array, gt_array = remove_empty_slices_sem_rins(
        #         imagem_especificada_array, gt_array)

        # centraliza as imagens
        # imagem_especificada_array, gt_array = centralizar_imagens(
        #     imagem_especificada_array, gt_array, width=width, height=height)

        # manter só os rins com lesões
        # imagem_especificada_array = manter_rins_com_lesoes(
        #     imagem_especificada_array, gt_array)

        new_mask = gt_array
        # Remove a máscara do rim, transforma em fundo
        new_mask = np.array(np.where(new_mask == 1, 1, new_mask), np.uint8)
        # Muda máscara da lesão de 2 para 1
        new_mask = np.array(np.where(new_mask == 2, 1, new_mask), np.uint8)

        # For necessário pois será salvo cada fatia indivudualmente
        for i in range(0, imagem_especificada_array.shape[0]):

            img_name = generate_case_filename_path(
                nome_paciente, nome_paciente, i, output_dir, is_mask=False, lua_pattern=False)

            np.savez_compressed(img_name, imagem_especificada_array[i])

        for i in range(0, new_mask.shape[0]):

            mask_name = generate_case_filename_path(
                nome_paciente, nome_paciente, i, output_dir, is_mask=True, lua_pattern=False)

            np.savez_compressed(mask_name, new_mask[i])


if __name__ == '__main__':
    SPECIFICATION_TEMPLATE = "imaging_7.nii.gz"

    width = 512
    height = 512

    dataset_type = ""

    # Dom
    # dataset_root_dir = r"D:\DOUTORADO_LUANA\bases2D\KiTS19_master"
    # Lua
    dataset_root_dir = r"D:\projeto-doutorado-luana\base\base_antiga\base_img"

    output_dir = r'D:\projeto-doutorado-luana\base\Kits19_npz_2D_so_rim_test'
    histogram_specification = True

    generate_imgs_masks_2d(dataset_root_dir, output_dir, dataset_type, width, height,
                           only_injuries_slices=False, histogram_specification=histogram_specification)
