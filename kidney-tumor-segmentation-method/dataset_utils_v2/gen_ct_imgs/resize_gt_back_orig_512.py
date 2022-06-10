import numpy as np

'''
Script para gerar as imagens das lesões com base na predição dos rins
'''

from dataset_utils_v2.gen_ct_imgs.methods.filters import *
from dataset_utils_v2.gen_ct_imgs.methods.center_images import *
from dataset_utils_v2.gen_ct_imgs.methods.retornar_regioes_rins import *

def resize_back(gt_prediction_array, resize_back_info):

    shape_before_delimit, shape_before_resize, original_shape, is_resized, positions = resize_back_info
    gt_prediction_array = np.where(gt_prediction_array == 1, 2, 0)

    # descentralizar as imagens
    gt_prediction_return = descentralizar_gt(
        gt_prediction_array, shape_before_resize)

    # redimensiona proporcionalmente para o original
    if(is_resized):
        # voltar ao tamanho antes do resize
        gt_prediction_return = resize_proporcional_image_2(
            gt_prediction_return, shape_before_delimit[2], shape_before_delimit[1])

    gt_prediction_return = back_image_original(
        gt_prediction_return, original_shape, positions).astype(np.uint8)

    gt_prediction_return = np.where(gt_prediction_return == 2, 1, 0)

    return gt_prediction_return
