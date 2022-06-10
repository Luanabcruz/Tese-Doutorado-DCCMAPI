
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
    from framework import metrics as m

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


# path_root = r'D:\DoutoradoLuana\codes\nii_luaExporter\reconstrucao\final'
path_root = r'D:\DoutoradoLuana\codes\pipeline_completo_random_v8\tumor_final_lower\segmentacao_final\lesao\descontinuosv2'
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
    # sitk_pred_array_k = post_processing_3D(sitk_pred_array_k)
    cm = confusion_matrix(gt_array_k.flatten(), sitk_pred_array_k.flatten(), labels=[0, 1])

    dice = m.metric_by_slice_cm(gt_array_k, sitk_pred_array_k,m.dice_cm,cm)
    iou = m.metric_by_slice_cm(gt_array_k, sitk_pred_array_k,m.jaccard_cm,cm)
    acc = m.metric_by_slice_cm(gt_array_k, sitk_pred_array_k,m.accuracy_cm,cm)
    sen = m.metric_by_slice_cm(gt_array_k, sitk_pred_array_k,m.sensitivity_cm,cm)
    spec = m.metric_by_slice_cm(gt_array_k, sitk_pred_array_k,m.specificity_cm,cm)
    # dices.append(dice)
    print("{},{},{},{},{},{}".format(case_name, dice,iou,acc,sen,spec))
    # print("{},{},".format(case_name, dice))

    # sitk_pred_array_k = sitk_pred_array_k.transpose(1, 2, 0 )
    
    # sitk_pred_array_k = np.where(sitk_pred_array_k == 1, 2, 0)

    # image2 = sitk.GetImageFromArray(np.asarray(sitk_pred_array_k, np.uint32))
    # im_k = sitk.ReadImage(os.path.join(
    #     kits19_path, case_name, "imaging.nii.gz"))
    
    # image2.CopyInformation(im_k)

    # sitk.WriteImage(image2, os.path.join(path_root, 'v2', pred))

# print("Mean: {:.4f}".format(np.asarray(dices).mean()))
