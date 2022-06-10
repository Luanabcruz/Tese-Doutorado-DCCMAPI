import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
'''
Script para gerar as imagens das lesões com base na predição dos rins
'''

import cv2
import glob

from methods.filters import *
from sklearn.metrics import confusion_matrix
from methods.center_images import *
from methods.retornar_regioes_rins import *

from utils_ds import create_nested_dir
from tqdm import tqdm
from resize_case import CaseResizeInfo


def dice_metric(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    tn, fp, fn, tp = cm.ravel()
    dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)
    return dice


def dice_by_case(y_true, y_pred):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]

    y_pred_new = []
    y_true_new = []

    for i in range(batch_size):
        if (np.sum(y_pred[i, :, :]) == 0) and (np.sum(y_true[i, :, :]) == 0):
            continue

        y_pred_new.append(y_pred[i, :, :])
        y_true_new.append(y_true[i, :, :])

    return dice_metric(np.asarray(y_true_new), np.asarray(y_pred_new))


def return_images(path_root, output_dir, save_nii=False):

    path_root = os.path.join(path_root, "")
    cont = 0
    media = 0
    folders = glob.glob(os.path.join(path_root, '*'))
    pred_kid_folder = './out_256_from_kidney/nii'

    test_cases = ["case_00009", "case_00011", "case_00012", "case_00018", "case_00024", "case_00025", "case_00028", "case_00041", "case_00043", "case_00045", "case_00056", "case_00066", "case_00076", "case_00078", "case_00082",
                  "case_00085", "case_00099", "case_00100", "case_00102", "case_00122", "case_00131", "case_00139", "case_00149", "case_00158", "case_00170", "case_00174", "case_00175", "case_00176", "case_00197", "case_00199", "case_00200"]

    cri = CaseResizeInfo()
    dices = []
    for folder in test_cases:

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        shape_before_delimit, shape_before_resize, original_shape, is_resized, positions = cri.load_case_scale_info(
            nome_paciente)

        if not nome_paciente in test_cases:
            continue

        folder += "\\"

        # print(nome_paciente)
        # print("paciente: ", cont)
        cont += 1
        entrou_rezise = False

        gt_prediction = sitk.ReadImage(os.path.join(
            pred_kid_folder, "prediction_{}".format(nome_paciente.replace("case_", ""))))
        # gt_prediction = sitk.ReadImage(folder+"prediction_00170.nii.gz")
        gt_prediction_array = sitk.GetArrayFromImage(gt_prediction)

        gt_prediction_array = gt_prediction_array.transpose(2, 0, 1)
        gt_prediction_array = np.where(gt_prediction_array == 3, 2, 0)

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

        gt_array = sitk.ReadImage(os.path.join(
            dataset_root_dir, nome_paciente, "segmentation.nii.gz"))
        gt_array = sitk.GetArrayFromImage(gt_array).transpose(2, 0, 1)

        gt_array = np.where(gt_array == 1, 0, gt_array).astype(np.uint8)

        # gt_prediction_return = sitk.GetImageFromArray(gt_prediction_return)
        # sitk.WriteImage(gt_prediction_return, "PREDICTION-RETORNO.nii")

        # gt_array = sitk.GetImageFromArray(gt_array)
        # sitk.WriteImage(gt_array, "GT.nii")

        dice = dice_by_case(gt_array, gt_prediction_return)
        dices.append(dice)
        print("{}: {:.4f}".format(nome_paciente, dice))
        # media = media + dice

        if save_nii:
            create_nested_dir(output_dir)
            gt_prediction_ = gt_prediction_return.transpose(1, 2, 0)
            gt_prediction_ = sitk.GetImageFromArray(gt_prediction_)

            sitk.WriteImage(gt_prediction_, output_dir +
                            "prediction_{}".format(nome_paciente.replace("case_", ""))+".nii.gz")

    print("Mean: {:.4f}".format(np.asarray(dices).mean()))
    # print(media/len(test_cases))


if __name__ == '__main__':

    # Dom
    # dataset_root_dir = r"D:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"
    # Lua
    # dataset_root_dir = r"D:\DOUTORADO_LUANA\bases2D\teste challenge\teste challenge"

    output_dir = r'./output_predict_nii_pred_dpn131_lesao_resized_512/'
    # dataset_root_dir = r"D:\Documentos\Projetos\teste\gt_predict"
    dataset_root_dir = r"D:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"

    return_images(dataset_root_dir, output_dir, save_nii=True)
