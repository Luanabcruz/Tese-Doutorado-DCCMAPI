import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

"""
Script para gerar as imagens das lesões com base na predição dos rins
"""

import cv2
import glob

from methods.filters import *
from methods.delimitar_regioes_rins import *
from methods.center_images import *

from utils import create_nested_dir


def generate_case_filename_path(
    dataset_type, case_name, index, output_dir, is_mask=False, lua_pattern=True
):

    file_template = ""

    if lua_pattern:
        if is_mask:
            path = os.path.join(output_dir, "masc_{}".format(dataset_type))
            file_template = "masc_{}-{}".format(case_name, index)
        else:
            path = os.path.join(output_dir, "img_{}".format(dataset_type))
            file_template = "{}-{}".format(case_name, index)
        # cria diretório caso não exista
        create_nested_dir(path)

        return os.path.join(path, file_template)

    else:

        if is_mask:
            path = os.path.join(output_dir, dataset_type, "Masks")
            file_template = "masc_{}-{}".format(case_name, index)
        else:
            path = os.path.join(output_dir, dataset_type, "Images")
            file_template = "{}-{}".format(case_name, index)
        # cria diretório caso não exista
        create_nested_dir(path)

        return os.path.join(path, file_template)


def generate_imgs_masks_2d(
    path_root,
    output_dir,
    width,
    height,
    only_injuries_slices=False,
    only_lesion_mask=False,
    histogram_specification=False,
    dilation_f=False,
):

    path_root = os.path.join(path_root, "")
    cont = 0
    folders = glob.glob(os.path.join(path_root, "*"))
    pred_kid_folder = "./output_predict_nii_25D_resunet101_balanced"

    test_cases = [
        "case_00009",
        "case_00011",
        "case_00012",
        "case_00018",
        "case_00024",
        "case_00025",
        "case_00028",
        "case_00041",
        "case_00043",
        "case_00045",
        "case_00056",
        "case_00066",
        "case_00076",
        "case_00078",
        "case_00082",
        "case_00085",
        "case_00099",
        "case_00100",
        "case_00102",
        "case_00122",
        "case_00131",
        "case_00139",
        "case_00149",
        "case_00158",
        "case_00170",
        "case_00174",
        "case_00175",
        "case_00176",
        "case_00197",
        "case_00199",
        "case_00200",
    ]

    for folder in folders:

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        if not nome_paciente in test_cases:
            continue

        folder += "\\"

        print(nome_paciente)
        print("paciente: ", cont)
        cont += 1

        imagem = sitk.ReadImage(folder + "imaging.nii.gz")
        imagem_array = sitk.GetArrayFromImage(imagem)

        gt_prediction = sitk.ReadImage(
            os.path.join(
                pred_kid_folder,
                "prediction_{}".format(nome_paciente.replace("case_", "")),
            )
        )
        # gt_prediction = sitk.ReadImage(folder+"segmentation.nii.gz")

        gt_prediction_array = sitk.GetArrayFromImage(gt_prediction)

        gt = sitk.ReadImage(folder + "segmentation.nii.gz")
        gt_array = sitk.GetArrayFromImage(gt)

        imagem_array = imagem_array.transpose(2, 0, 1)
        gt_prediction_array = gt_prediction_array.transpose(2, 0, 1)
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
            imagem_especificada_array, rgb_scale=False
        )

        (
            imagem_original_bb_array,
            gt_original_bb_array,
            imagem_prediction_bb_array,
            gt_prediction_bb_array,
        ) = delimite_predict_origin(
            imagem_especificada_array, gt_array, gt_prediction_array
        )

        # manter só os rins com lesões // mantém a região do gt dilatado
        imagem_especificada_array = manter_rins_com_lesoes(
            imagem_original_bb_array, gt_prediction_bb_array
        )

        # só pra não precisar mudar os nomes posteriormente
        gt_array = gt_original_bb_array.copy()

        # redimensiona proporcionalmente
        if gt_array.shape[1] > width or gt_array.shape[2] > height:
            print("ENTROU RESIZE!")
            imagem_especificada_array = resize_proporcional_image(
                imagem_especificada_array.copy(), width, height
            )

            gt_array = resize_proporcional_image(gt_array, width, height)

            gt_prediction_bb_array = resize_proporcional_image(
                np.asarray(gt_prediction_bb_array, np.uint8), width, height)

        # centraliza as imagens
        imagem_especificada_array, gt_prediction_bb_array = centralizar_imagens(
            imagem_especificada_array, gt_prediction_bb_array
        )

        new_mask = gt_prediction_bb_array.transpose(1, 2, 0)

        mask_itk = sitk.GetImageFromArray(new_mask)
        create_nested_dir(output_dir)
        sitk.WriteImage(mask_itk, os.path.join(
            output_dir, nome_paciente + ".nii.gz"))


if __name__ == "__main__":
    SPECIFICATION_TEMPLATE = "imaging_176.nii.gz"

    width = 256
    height = 256

    # Dom
    dataset_root_dir = r"D:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"
    # Lua
    # dataset_root_dir = r"D:\DOUTORADO_LUANA\bases2D\teste challenge\teste challenge"

    output_dir = r"./output_pred_kidney_256"

    generate_imgs_masks_2d(
        dataset_root_dir,
        output_dir,
        width,
        height,
        only_lesion_mask=True,
        histogram_specification=False,
    )
