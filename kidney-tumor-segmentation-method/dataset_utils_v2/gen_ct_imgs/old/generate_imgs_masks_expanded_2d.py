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
            file_template = 'masc_{}-{}.png'.format(case_name, index)
        else:
            path = os.path.join(output_dir, 'img_{}'.format(dataset_type))
            file_template = '{}-{}.png'.format(case_name, index)
        # cria diretório caso não exista
        create_nested_dir(path)

        return os.path.join(path, file_template)

    else:

        if is_mask:
            path = os.path.join(output_dir, dataset_type, 'Masks')
            file_template = 'masc_{}-{}.png'.format(case_name, index)
        else:
            path = os.path.join(output_dir, dataset_type, 'Images')
            file_template = '{}-{}.png'.format(case_name, index)
        # cria diretório caso não exista
        create_nested_dir(path)

        return os.path.join(path, file_template)


def generate_imgs_masks_2d(path_root, output_dir, local, width, height, only_injuries_slices=False, image_format=True, only_lesion_mask=False, histogram_specification=False):

    path_root = os.path.join(path_root, local)

    cont = 0
    folders = glob.glob(os.path.join(path_root, '*'))

    for folder in folders:

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        folder += "\\"

        print(nome_paciente)
        print("paciente: ", cont)
        cont += 1

        imagem = sitk.ReadImage(folder+"imaging.nii.gz")
        imagem_array = sitk.GetArrayFromImage(imagem)

        gt = sitk.ReadImage(folder+"segmentation.nii.gz", sitk.sitkUInt8)
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

        if image_format:
            imagem_especificada_array = rescale_intensity(
                imagem_especificada_array)
        else:
            imagem_especificada_array = rescale_intensity(
                imagem_especificada_array, rgb_scale=False)

        # dilatação dos rins
        gt_dilatation = np.array(np.where(gt_array > 0, 1, 0), np.uint8)
        gt_dilatation = sitk.GetImageFromArray(gt_dilatation)
        gt_dilated = dilation_filter(gt_dilatation, kernel_value=3)
        # fim da dilatação dos rins

        # Coloca as imgs e gts (especialista) do mesmo tamanho
        imagem_original_bb_array, gt_original_bb_array, imagem_dilated_bb_array, gt_dilated_bb_array = delimite_predict_origin(
            imagem_especificada_array, gt_array, gt_dilated)

        # manter só os rins com lesões // mantém a região do gt dilatado
        imagem_especificada_array = manter_rins_com_lesoes(
            imagem_original_bb_array, gt_dilated_bb_array)

        # só pra não precisar mudar os nomes posteriormente
        gt_array = gt_original_bb_array.copy()

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
            imagem_especificada_array, gt_array = remove_empty_slices(
                imagem_especificada_array, gt_array)

        # centraliza as imagens
        imagem_especificada_array, gt_array = centralizar_imagens(
            imagem_especificada_array, gt_array)

        new_mask = gt_array

        if only_lesion_mask:
            new_mask = np.array(np.where(new_mask == 2, 255, 0), np.uint8)
        else:
            new_mask = np.array(
                np.where(new_mask == 1, 255, new_mask), np.uint8)
            new_mask = np.array(
                np.where(new_mask == 2, 128, new_mask), np.uint8)

        for i in range(imagem_especificada_array.shape[0]):
            img_name = generate_case_filename_path(
                local, nome_paciente, i, output_dir, is_mask=False, lua_pattern=False)
            mask_name = generate_case_filename_path(
                local, nome_paciente, i, output_dir, is_mask=True, lua_pattern=False)

            if image_format:

                cv2.imwrite(img_name, imagem_especificada_array[i])
                cv2.imwrite(mask_name, new_mask[i])
            else:
                np.save(img_name.replace('.png', ''),
                        imagem_especificada_array[i])
                np.save(mask_name.replace('.png', ''), new_mask[i])


if __name__ == '__main__':
    SPECIFICATION_TEMPLATE = "./kidney_dataset_utils/imaging_26.nii.gz"

    width = 256
    height = 256
    dataset_types = ["treino", "validacao"]
    # dataset_types = ["teste"]
    # Dom
    dataset_root_dir = r"D:\DOUTORADO_LUANA\KiTS19_lua"
    # Lua
    # dataset_root_dir = r"D:\Documentos\Projetos\projeto-doutorado\base"

    output_dir = 'kits19_picklenumpy_2_classes_expand'
    histogram_specification = False
    only_lesion_mask = True
    only_injuries_slices = True
    image_format = False

    for dataset_type in dataset_types:
        generate_imgs_masks_2d(dataset_root_dir, output_dir, dataset_type, width, height,
                               only_injuries_slices=only_injuries_slices, image_format=image_format, only_lesion_mask=only_lesion_mask, histogram_specification=histogram_specification)
