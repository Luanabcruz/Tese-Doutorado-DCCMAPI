import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
'''
Script para gerar as imagens das lesões com base na predição dos rins
'''

import cv2
import glob

from methods.filters import *
#from methods.delimitar_regioes_rins import *
from methods.center_images import *
from methods.retornar_regioes_rins import *

from utils_ds import create_nested_dir
from resize_case import CaseResizeInfo


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
    pred_kid_folder = r'D:\DoutoradoLuana\codes\pipeine_completo_nii_sem_balanc\kidney\nii'

    test_cases = ["case_00009", "case_00011", "case_00012", "case_00018", "case_00024", "case_00025", "case_00028", "case_00041", "case_00043", "case_00045", "case_00056", "case_00066", "case_00076", "case_00078", "case_00082",
                  "case_00085", "case_00099", "case_00100", "case_00102", "case_00122", "case_00131", "case_00139", "case_00149", "case_00158", "case_00170", "case_00174", "case_00175", "case_00176", "case_00197", "case_00199", "case_00200"]

    cri = CaseResizeInfo()
    # test_cases = test_cases[:4]
    for folder in folders:

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        if not nome_paciente in test_cases:
            continue

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

        gt = sitk.ReadImage(folder+"segmentation.nii.gz")
        gt_array = sitk.GetArrayFromImage(gt)

        imagem_array = imagem_array.transpose(2, 0, 1)
        gt_prediction_array = gt_prediction_array.transpose(2, 0, 1)
        gt_array = gt_array.transpose(2, 0, 1)
        original_shape = gt_array.shape

        
        imagem_especificada_array = windowing(imagem_array)

        imagem_especificada_array = rescale_intensity(
            imagem_especificada_array, rgb_scale=False)

        positions, imagem_original_bb_array, gt_original_bb_array, imagem_prediction_bb_array, gt_prediction_bb_array = delimite_predict_origin(
            imagem_especificada_array, gt_array, gt_prediction_array)

        # manter só os rins com lesões // mantém a região do gt dilatado
        imagem_especificada_array = manter_fatias_com_rins(
            imagem_original_bb_array, gt_prediction_bb_array)

        # só pra não precisar mudar os nomes posteriormente
        gt_array = gt_original_bb_array.copy()

        # redimensiona proporcionalmente
        if(gt_array.shape[1] > width or gt_array.shape[2] > height):
            print("ENTROU RESIZE!")
            is_resized = True
            imagem_especificada_array = resize_proporcional_image(
                imagem_especificada_array, width, height)
            gt_array = resize_proporcional_image(gt_array, width, height)

        # seria isso?
        shape_before_resize = gt_array.shape

        # centraliza as imagens
        imagem_especificada_array, gt_array = centralizar_imagens(
            imagem_especificada_array, gt_array)

        cri.save_case_scale_info(
            nome_paciente, gt_prediction_bb_array.shape, shape_before_resize, original_shape, positions)

        new_mask = gt_array
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
    dataset_root_dir = r"E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"
    # Lua
    # dataset_root_dir = r"D:\DOUTORADO_LUANA\bases2D\teste challenge\teste challenge"

    output_dir = r'D:\DoutoradoLuana\datasets\gerados\sem_balanceamento\pred_resunet25D_sem_balanc_256'

    generate_imgs_masks_2d(dataset_root_dir, output_dir,
                           width, height, only_lesion_mask=True, histogram_specification=False)
