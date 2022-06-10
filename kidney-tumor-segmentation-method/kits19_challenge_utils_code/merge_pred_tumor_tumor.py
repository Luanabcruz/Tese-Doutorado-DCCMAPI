import os
import SimpleITK as sitk
import numpy as np

tumor_pred_resunet = r'C:\Users\usuario\Desktop\saida'
tumor_pred_dpn = r'E:\DoutoradoLuana\datasets\nii\pred_TW_V3_0_deep_plus_dpn131\output_predict_nii'
output = r'C:\Users\usuario\Desktop\teste_challenge_resunet101_dpn131_com_pos'

for pred in os.listdir(tumor_pred_resunet):
    print(pred)

    sitk_pred_k = sitk.ReadImage(os.path.join(tumor_pred_resunet, pred))
    sitk_pred_array_k = sitk.GetArrayFromImage(sitk_pred_k)

    sitk_pred_t = sitk.ReadImage(os.path.join(tumor_pred_dpn, pred))
    sitk_pred_array_t = sitk.GetArrayFromImage(sitk_pred_t)

    mask_final = np.bitwise_or(sitk_pred_array_k, sitk_pred_array_t)
    # mask_final = np.where(mask_final == 3, 2, mask_final)

    mask_itk = sitk.GetImageFromArray(np.asarray(mask_final, np.uint32))
    mask_itk.CopyInformation(sitk_pred_k)
    sitk.WriteImage(mask_itk, os.path.join(
        output, pred))
