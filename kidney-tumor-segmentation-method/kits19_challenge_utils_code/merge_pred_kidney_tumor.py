import os
import SimpleITK as sitk
import numpy as np
kidney_pred = r'C:\Users\usuario\Desktop\teste_challenge_rim_sem_pos\output_predict_nii'
tumor_pred = r'C:\Users\usuario\Desktop\teste_challenge_lesao_sem_pos\output_predict_nii'
output = r'C:\Users\usuario\Desktop\teste_challenge_rim_lesao_sem_pos'

for pred in os.listdir(kidney_pred):
    print(pred)

    sitk_pred_k = sitk.ReadImage(os.path.join(kidney_pred, pred))
    sitk_pred_array_k = sitk.GetArrayFromImage(sitk_pred_k)

    sitk_pred_t = sitk.ReadImage(os.path.join(tumor_pred, pred))
    sitk_pred_array_t = sitk.GetArrayFromImage(sitk_pred_t)

    mask_final = np.bitwise_xor(sitk_pred_array_k, sitk_pred_array_t)
    mask_final = np.where(mask_final == 3, 2, mask_final)

    mask_itk = sitk.GetImageFromArray(np.asarray(mask_final, np.uint32))
    mask_itk.CopyInformation(sitk_pred_k)
    sitk.WriteImage(mask_itk, os.path.join(
        output, pred))
