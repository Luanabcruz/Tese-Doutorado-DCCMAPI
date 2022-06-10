from dataset import load_case_mask_folder

from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
cases_test = [9, 11, 18, 26, 28, 41, 43, 45, 66, 76, 78, 82, 85, 97, 100, 103,
              122, 129, 131, 139, 149, 158, 160, 165, 170, 174, 176, 191, 197, 199, 200]


def dice_metric(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)
    return dice


gt_path = r"C:\Users\usuario\Documents\UFMA\DoutoradoLuana\segmentacao2d\datasets\imagens_pred97_lua2307\teste\Masks"
pred_path = r'C:\Users\usuario\Desktop\predicao_max_97rim'

# for case_code in cases_codes:
dices = []
for case_code in cases_test:
    # essa funcao carrega todas as fatias de um caso em um unico array. certo? sei não k so to explicando. Eh o que essa funcao faz. pego o numero do teste e vou deixar todas no mesmo array para calcular o diceentendi
    gt_case = load_case_mask_folder(gt_path, case_code)
    pred_case = load_case_mask_folder(pred_path, case_code)

    dice = dice_metric(gt_case, pred_case)
    dices.append(dice)
    # print('Caso {} => Max dice: {}'.format(case_code, dice))
    # print('___________________')
    print('{},{}'.format(case_code, dice))


print('___________________')
dices = np.asarray(dices)
print("Dice: {} ± {} ".format(dices.mean(), dices.std()))
