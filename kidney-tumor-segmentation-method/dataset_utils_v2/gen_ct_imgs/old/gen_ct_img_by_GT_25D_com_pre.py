'''
Script para gerar as imagens das lesões com base na marcação do rim pelo especialista
'''


import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import cv2
import glob

from methods.filters import *
from methods.delimitar_regioes_rins import *
from methods.center_images import *

# from utils import create_nested_dir
import time


def bilateral2d_filter(image):

    image = np.asarray(image, np.float32)

    for i in range(0, image.shape[0]):
        image[i, :, :] = cv2.bilateralFilter(image[i, :, :], 4, 35, 35)
    return image


def filter2D(image):
    kernel = (3, 3)
    image = np.asarray(image, np.float32)

    for i in range(0, image.shape[0]):
        image[i, :, :] = cv2.filter2D(image[i, :, :], -1, kernel)
    return image


def medianBlur(image):
    kernel = (3, 3)
    image = np.asarray(image, np.float32)

    for i in range(0, image.shape[0]):
        image = np.asarray(image, np.float32)
        image[i, :, :] = cv2.medianBlur(image[i, :, :], kernel[0])

    return image


def topHat(image):
    kernel = (3, 3)

    filterSize = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       filterSize)

    for i in range(0, image.shape[0]):
        image = np.asarray(image, np.float32)
        image[i, :, :] += cv2.morphologyEx(
            image[i, :, :], cv2.MORPH_TOPHAT, kernel)
        image[i, :, :] = image[i, :, :]/image.max()
    print(image.max())
    return image


def show(image, gt_array):

    for i in range(0, image.shape[0]):
        plt.gcf().canvas.set_window_title(str(i+1))
        if gt_array[i].max() > 0:
            plt.imshow(image[i, :, :], cmap='gray')
            plt.show()


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
        since = time.time()
        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        if (nome_paciente != "case_00184"):
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

        imagem_array = windowing(imagem_array)

        # print("Aplicando o bilateral 2D")
        # imagem_array = bilateral2d_filter(imagem_array)
        # print("Aplicando o histograma adaptativo")
        # preimage = sitk.GetImageFromArray(imagem_array)
        # preimage = histAdapt(preimage)
        # imagem_array = sitk.GetArrayFromImage(preimage)

        # print("Aplicando o histograma eql")
        # preprocessing_image = eqHist(preprocessing_image)
        # print("Pegando o array...")
        # imagem_especificada_array = sitk.GetArrayFromImage(preprocessing_image)
        # preprocessing_image = None
        # print("Janelamento...")

        # imagem_especificada_array = windowing(imagem_array)

        imagem_especificada_array = rescale_intensity(
            imagem_array, rgb_scale=False)

        print('Aplicando filter 2D')
        imagem_especificada_array = filter2D(imagem_especificada_array)
        print('Aplicando filter bilateral')
        imagem_especificada_array = bilateral2d_filter(
            imagem_especificada_array)
        print('Median Blur')
        imagem_especificada_array = medianBlur(imagem_especificada_array)
        print('Tophat')
        imagem_especificada_array = topHat(imagem_especificada_array)

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

        if True:
            # retira fatias que não possuem lesão
            imagem_especificada_array, gt_array = remove_empty_slices_sem_lesoes(
                imagem_especificada_array, gt_array)
        else:
            # para retirar as fatias que não tem rins
            imagem_especificada_array, gt_array = remove_empty_slices(
                imagem_especificada_array, gt_array)

        # centraliza as imagens
        imagem_especificada_array, gt_array = centralizar_imagens(
            imagem_especificada_array, gt_array, width=width, height=height)

        # manter só os rins com lesões
        imagem_especificada_array = manter_rins_com_lesoes(
            imagem_especificada_array, gt_array)

        show(imagem_especificada_array, gt_array)
        exit()

        new_mask = gt_array
        # Remove a máscara do rim, transforma em fundo
        new_mask = np.array(np.where(new_mask == 1, 0, new_mask), np.uint8)
        # Muda máscara da lesão de 2 para 1
        new_mask = np.array(np.where(new_mask == 2, 1, new_mask), np.uint8)

        empty_image = np.zeros([width, height], dtype=np.uint8)
        channel = 3

        # se canal for 5 tem que se adicionar 2 imagens no inicio, e 2 no fim. Se for 3 uma no inicio e uma no fim resolve.
        imagem_especificada_array = np.insert(
            imagem_especificada_array, 0, empty_image, axis=0)

        imagem_especificada_array = np.concatenate(
            (imagem_especificada_array, np.expand_dims(empty_image, axis=0)), axis=0)

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

        time_elapsed = time.time() - since
        print('Generate case complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    SPECIFICATION_TEMPLATE = "kidney_dataset_utils/imaging_176.nii.gz"

    width = 256
    height = 256

    dataset_type = ""

    # Dom
    # dataset_root_dir = r"D:\DOUTORADO_LUANA\bases2D\KiTS19_master"
    # Lua
    dataset_root_dir = r"D:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"

    output_dir = r'D:\trash'
    histogram_specification = False

    generate_imgs_masks_2d(dataset_root_dir, output_dir, dataset_type, width, height,
                           only_injuries_slices=False, histogram_specification=histogram_specification)
