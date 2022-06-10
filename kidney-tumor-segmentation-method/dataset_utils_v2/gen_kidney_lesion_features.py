'''
SCRIPT para gerar as features de cada caso para análise de distribuição no treino/validação
'''

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import cv2

from gen_ct_imgs.methods.filters import *
from gen_ct_imgs.methods.kidneys_lesions_features_inf import *

from utils import create_nested_dir

import csv
import os


def save_case_info(cases_info=None):

    fieldnames = ['case', 'kidney_area', 'kidney_position', 'tumors_num']

    # adiciona campos do tumor
    fieldnames_tumor = ['tumor{}_area', 'tumor{}_color', 'tumor{}_num_slices']

    for index in range(1, 23):
        for f_tumor in fieldnames_tumor:
            fieldnames.append(f_tumor.format(index))

    filename = 'cases_info_270720.csv'
    if cases_info == None:
        with open(os.path.join('.', filename), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    else:
        with open(os.path.join('.', filename), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(cases_info)


def create_case_info_file():
    # cria o arquivo  e define os cabeçalhos do csv
    save_case_info(None)


def add_case_info_file(cases_info):
    # salva dicionário com as mesmas chaves que foram definidos no cabeçalho
    save_case_info(cases_info)


def generate_imgs_masks_2d(path_root, output_dir, local, width, height, only_injuries_slices=False, histogram_specification=False):

    path_root = os.path.join(path_root, local)

    cont = 0
    folders = glob.glob(os.path.join(path_root, '*'))

    quant_lesion = -1

    for folder in folders:

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        folder += "\\"
        # if nome_paciente != 'case_00097':
        #     continue

        print(nome_paciente)
        print("paciente: ", cont)
        cont += 1

        imagem = sitk.ReadImage(folder+"imaging.nii.gz")
        imagem_array = sitk.GetArrayFromImage(imagem)

        gt = sitk.ReadImage(folder+"segmentation.nii.gz")
        gt_array = sitk.GetArrayFromImage(gt)

        imagem_array = imagem_array.transpose(2, 0, 1)
        gt_array = gt_array.transpose(2, 0, 1)

        # posição 0 - esquerdo, 1 - direito
        labels_kidneys = labels_kidney(gt_array)
        labels_kidneys = np.unique(labels_kidneys).tolist()

        imagem_array = rescale_intensity(
            imagem_array)

        for (key, label_k) in enumerate(labels_kidneys):

            gt_just_a_kidney, kidney_area, kidney_slices = kidney_features(
                gt_array, label_k)
            #print("Área do rim: ", kidney_area)
            #print("Quant. fatias rim: ", kidney_slices)

            case_info = {}
            case_info['case'] = nome_paciente
            case_info['kidney_area'] = kidney_area
            case_info['kidney_position'] = 'left' if key == 0 else 'right'

            if(np.amax(gt_just_a_kidney) > 1):
                print("Tem lesão")
                labels_lesion = quantity_labels_lesion(gt_just_a_kidney)
                case_info['tumors_num'] = len(labels_lesion)

                for (key_l, label_l) in enumerate(labels_lesion):
                    lesion_area, lesion_slices, color_result = lesion_features(
                        imagem_array, gt_just_a_kidney, label_l)

                    case_info['tumor{}_area'.format(key_l+1)] = lesion_area
                    case_info['tumor{}_num_slices'.format(
                        key_l+1)] = lesion_slices
                    case_info['tumor{}_color'.format(
                        key_l+1)] = 'dark' if color_result == 0 else 'light'

                    #print("Área da lesão: ", lesion_area)
                    #print("Quant. fatias lesão: ", lesion_slices)
            else:
                print("Não tem lesão")

            save_case_info(case_info)


if __name__ == '__main__':
    SPECIFICATION_TEMPLATE = "imaging_176.nii.gz"

    width = 256
    height = 256

    dataset_types = ["treino", "validacao", "teste"]
    # dataset_types = ["teste"]
    # Dom
    dataset_root_dir = r"D:\DOUTORADO_LUANA\KiTS19_lua"
    # Lua
    # dataset_root_dir = r"D:\Documentos\Projetos\projeto-doutorado\base"

    output_dir = 'imagens_master_3_classes_lesao'
    histogram_specification = False

    # cria o arquivo  de casos
    # isso aqui eh o qyue tava sobrescrevendo
    create_case_info_file()

    for dataset_type in dataset_types:
        if dataset_type == 'teste':
            generate_imgs_masks_2d(dataset_root_dir, output_dir, dataset_type, width, height,
                                   only_injuries_slices=True, histogram_specification=histogram_specification)
        else:
            generate_imgs_masks_2d(dataset_root_dir, output_dir, dataset_type, width, height,
                                   only_injuries_slices=True, histogram_specification=histogram_specification)
