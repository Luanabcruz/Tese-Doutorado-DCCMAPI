
import torch
from kidney_utils.dataset import load_case_mask_folder

from kidney_utils.metrics import mean_dice_coef_remove_empty, dice_by_case
import numpy as np


'''
    Esse código foi feito para funcionar com as imagens das fatias do kits19.
    Cada caso é formato por um conjunto de fatias. É realizado uma predição em cada fatia, e essas predições são agrupadas para que sejam aplicadas as métricas em cada caso.
    No fim, calcula-se a média e desvio padrão das métricas preditas em cada caso.
'''

# Representa a lista de casos de testes a serem agrupados primeiro e depois preditos
cases_test = [9, 11, 18, 26, 28, 41, 43, 45, 66, 76, 78, 82, 85, 97, 100, 103,
              122, 129, 131, 139, 149, 158, 160, 165, 170, 174, 176, 191, 197, 199, 200]

gt_path = r'C:\Users\usuario\Desktop\lua\novo\Masks_gt'
pred_path = r'C:\Users\usuario\Desktop\lua\novo\Masks_pred97'

dices = []


for case_code in cases_test:

    case_gt = load_case_mask_folder(gt_path, case_code)
    case_pred = load_case_mask_folder(pred_path, case_code)

    dice = mean_dice_coef_remove_empty(case_pred, case_gt)
    print('Case code {} => Dice: '.format(case_code), dice)
    dices.append(dice)

dices = np.asarray(dices)
print('dice medio: {} (std: {})'.format(dices.mean(), dices.std()))
# print('jaccard: {} (std: {})'.format(jaccard.mean(), jaccard.std()))
# print('Excel version')
# print('dice: =ROUND({},4)&"±"&ROUND({},2)'.format(dice.mean(), dice.std()))
# print('jaccard: =ROUND({},4)&"±"&ROUND({},2)'.format(
#     jaccard.mean(), jaccard.std()))
