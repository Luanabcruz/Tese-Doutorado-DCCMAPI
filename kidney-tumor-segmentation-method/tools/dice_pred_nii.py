
import os
import SimpleITK as sitk
import numpy as np
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from kidney_utils.post_process import post_processing_3d_two_biggest_elem, post_processing_3D, post_processing_3D_morphological_closing


def dice_metric(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1])
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


path_root = r'D:\DoutoradoLuana\RESULTADOS_TESE\F13\seg_final_tumor\segmentacao_final\lesao\descontinuosv2'
kits19_path =r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'

dices = []

for pred in os.listdir(path_root):

    sitk_pred_k = sitk.ReadImage(os.path.join(path_root, pred))
    sitk_pred_array_k = sitk.GetArrayFromImage(sitk_pred_k)

    sitk_pred_array_k = np.where(sitk_pred_array_k == 1, 0, sitk_pred_array_k)
    sitk_pred_array_k = np.where(sitk_pred_array_k == 2, 1, sitk_pred_array_k)
    sitk_pred_array_k = sitk_pred_array_k.transpose(2, 0, 1)

    case_name = "case_"+pred.replace("prediction_", "").replace(".nii.gz", "")

    gt_k = sitk.ReadImage(os.path.join(
        kits19_path, case_name, "segmentation.nii.gz"))
    gt_array_k = sitk.GetArrayFromImage(gt_k)
    gt_array_k = np.where(gt_array_k == 1, 0, gt_array_k)
    gt_array_k = np.where(gt_array_k == 2, 1, gt_array_k)

    gt_array_k = gt_array_k.transpose(2, 0, 1)
    sitk_pred_array_k = post_processing_3d_two_biggest_elem(sitk_pred_array_k)
    dice = dice_by_case(gt_array_k, sitk_pred_array_k)
    dices.append(dice)
    print("{}: {:.4f}".format(case_name, dice))

print("Mean: {:.4f}".format(np.asarray(dices).mean()))
