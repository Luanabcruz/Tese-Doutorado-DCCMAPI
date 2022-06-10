import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
'''
Script para gerar as imagens das lesões com base na predição dos rins
'''

import cv2
import glob

from dataset_utils_v2.gen_ct_imgs.methods.filters import *
from dataset_utils_v2.gen_ct_imgs.methods.center_images import *
from dataset_utils_v2.gen_ct_imgs.methods.retornar_regioes_rins import *

from dataset_utils_v2.gen_ct_imgs.utils_ds import create_nested_dir
from dataset_utils_v2.gen_ct_imgs.resize_case import CaseResizeInfo


def generate_tumor_image_mask(pred_kid_folder, cases_info_dir, case_name, config_resize_filename='config_resize.json'):

    width = 256
    height = 256

    cri = CaseResizeInfo(filename=config_resize_filename)
    
    imagem = sitk.ReadImage(os.path.join(cases_info_dir, case_name,"imaging.nii.gz"))
    imagem_array = sitk.GetArrayFromImage(imagem)

    gt_prediction = sitk.ReadImage(os.path.join(
        pred_kid_folder, "prediction_{}.nii.gz".format(case_name.replace("case_", ""))))
    
    gt_prediction_array = sitk.GetArrayFromImage(gt_prediction)

    gt = sitk.ReadImage(os.path.join(cases_info_dir, case_name,"segmentation.nii.gz"))
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
        is_resized = True
        imagem_especificada_array = resize_proporcional_image(
            imagem_especificada_array, width, height)
        gt_array = resize_proporcional_image(gt_array, width, height)

    # seria isso?
    shape_before_resize = gt_array.shape

    # centraliza as imagens
    imagem_especificada_array, gt_array = centralizar_imagens(
        imagem_especificada_array, gt_array)

    info = cri.save_case_scale_info(
        case_name, gt_prediction_bb_array.shape, shape_before_resize, original_shape, positions)

    new_mask = gt_array
    new_mask = np.array(np.where(new_mask == 1, 0, new_mask), np.uint8)
    new_mask = np.array(np.where(new_mask == 2, 1, new_mask), np.uint8)

    empty_image = np.zeros([width, height], dtype=np.uint8)
    channel = 3

    imagem_especificada_array = np.insert(
        imagem_especificada_array, 0, empty_image, axis=0)

    imagem_especificada_array = np.concatenate(
        (imagem_especificada_array, np.expand_dims(empty_image, axis=0)), axis=0)

    all_new_vol = None
    for i in range(0, imagem_especificada_array.shape[0]-channel+1):
        new_vol = None
        for j in range(0, channel):
            if new_vol is None:
                new_vol = np.expand_dims(
                    np.copy(imagem_especificada_array[i+j]), axis=0)
            else:
                new_vol = np.concatenate(
                    (new_vol, np.expand_dims(imagem_especificada_array[i+j], axis=0)), axis=0)

        if all_new_vol is None:
                all_new_vol = np.expand_dims(
                    np.copy(new_vol), axis=0)
        else:
            all_new_vol = np.concatenate(
                (all_new_vol, np.expand_dims(new_vol, axis=0)), axis=0)

    
    case = {}

    case['image'] = np.expand_dims(all_new_vol, axis=1)
    case['mask'] = new_mask
    case['info'] = info

    return case


