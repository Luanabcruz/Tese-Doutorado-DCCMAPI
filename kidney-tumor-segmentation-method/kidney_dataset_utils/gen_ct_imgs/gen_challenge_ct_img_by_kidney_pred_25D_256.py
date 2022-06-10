import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
'''
Script para gerar as imagens das lesões com base na predição dos rins do challenge
'''

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


def generate_imgs_masks_2d(path_root, output_dir, width, height, only_injuries_slices=False, only_lesion_mask=False, histogram_specification=False, dilation_f=False):

    path_root = os.path.join(path_root, "")
    cont = 0
    folders = glob.glob(os.path.join(path_root, '*'))
    pred_kid_folder = './output_predict_nii_25D_resunet101'

    cri = CaseResizeInfo(filename='cases_scale_info_challenge.json')              
    
    for folder in folders:

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        folder += "\\"

        print(nome_paciente)
        print("paciente: ", cont)
        cont += 1

        imagem = sitk.ReadImage(folder+"imaging.nii.gz")
        imagem_array = sitk.GetArrayFromImage(imagem)

        gt_prediction = sitk.ReadImage(os.path.join(
            pred_kid_folder, "prediction_{}".format(nome_paciente.replace("case_", ""))))
        # gt_prediction = sitk.ReadImage(folder+"segmentation.nii.gz")

        gt_prediction_array = sitk.GetArrayFromImage(gt_prediction)

        imagem_array = imagem_array.transpose(2, 0, 1)
        gt_prediction_array = gt_prediction_array.transpose(2, 0, 1)
        original_shape = imagem_array.shape

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

        positions, gt_prediction_bb_array, img_prediction_bb_array = function_delimited_kidneys_challenge(
            gt_prediction_array, imagem_especificada_array)

        # manter só os rins com lesões
        imagem_especificada_array = manter_rins_com_lesoes(
            img_prediction_bb_array, gt_prediction_bb_array)

        # redimensiona proporcionalmente
        if(gt_prediction_bb_array.shape[1] > width or gt_prediction_bb_array.shape[2] > height):
            print("ENTROU RESIZE!")
            imagem_especificada_array = resize_proporcional_image(
                imagem_especificada_array, width, height)
            gt_prediction_bb_array = resize_proporcional_image(gt_prediction_bb_array, width, height)

        # seria isso?
        shape_before_resize = gt_array.shape

        # centraliza as imagens
        imagem_especificada_array, gt_prediction_bb_array = centralizar_imagens(
            imagem_especificada_array, gt_prediction_bb_array)

        cri.save_case_scale_info(
            nome_paciente, gt_prediction_bb_array.shape, shape_before_resize, original_shape, positions)

        new_mask = gt_prediction_bb_array
        new_mask = np.array(np.where(new_mask == 1, 0, new_mask), np.uint8)
        new_mask = np.array(np.where(new_mask == 2, 1, new_mask), np.uint8)

        empty_image = np.zeros([width, height], dtype=np.uint8)
        channel = 3

        # se canal for 5 tem que se adicionar 2 imagens no inicio, e 2 no fim. Se for 3 uma no inicio e uma no fim resolve.
        imagem_especificada_array = np.insert(
            imagem_especificada_array, 0, empty_image, axis=0)

        # imagem_especificada_array = np.insert(
        #     imagem_especificada_array, 0, empty_image.copy(), axis=0)

        imagem_especificada_array = np.concatenate(
            (imagem_especificada_array, np.expand_dims(empty_image, axis=0)), axis=0)

        # imagem_especificada_array = np.concatenate(
        #     (imagem_especificada_array, np.expand_dims(empty_image.copy(), axis=0)), axis=0)

        for i in range(0, imagem_especificada_array.shape[0]-channel+1):
            img_name = generate_case_filename_path(
                nome_paciente, nome_paciente, i, output_dir, is_mask=False, lua_pattern=False)

            new_vol = None

            for j in range(0, channel):
                if new_vol is None:
                    new_vol = np.expand_dims(
                        np.copy(imagem_especificada_array[i+j]), axis=0)
                else:
                    new_vol = np.concatenate(
                        (new_vol, np.expand_dims(imagem_especificada_array[i+j], axis=0)), axis=0)

            np.savez_compressed(img_name, new_vol)

        for i in range(0, new_mask.shape[0]):

            mask_name = generate_case_filename_path(
                nome_paciente, nome_paciente, i, output_dir, is_mask=True, lua_pattern=False)

            np.savez_compressed(mask_name, new_mask[i])


if __name__ == '__main__':
    SPECIFICATION_TEMPLATE = "imaging_176.nii.gz"

    width = 256
    height = 256

    # Dom
    dataset_root_dir = r"D:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"
    # Lua
    # dataset_root_dir = r"D:\DOUTORADO_LUANA\bases2D\teste challenge\teste challenge"

    output_dir = r'E:\DoutoradoLuana\datasets\gerados\pred_resunet25D_balanceado_256'

    generate_imgs_masks_2d(dataset_root_dir, output_dir,
                           width, height, only_lesion_mask=True, histogram_specification=False)
