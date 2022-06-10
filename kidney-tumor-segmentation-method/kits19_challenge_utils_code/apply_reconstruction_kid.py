

import sys
import numpy as np
import SimpleITK as sitk
import os
import pathlib

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from kidney_utils.post_process import post_processing_3d_two_biggest_elem, post_processing_3D


kidney_pred = r'../pipeline_completo_nii/nii_pred_resunet101_rim'
tumor_raw_pred = r'../pipeline_completo_nii/nii_pred_resunet101_reconstrucao_lesao'

tumor_pred = r'../pipeline_completo_nii/nii_pred_dpn131_lesao_resized_512'  # reconstrucao

output = r'../pipeline_completo_nii/nii_pipeline_completo_rim_final'


for pred in os.listdir(kidney_pred):
    print(pred)

    # ETAPA 1 - juntar rim com reconstrução

    # PREDIÇÃO DO RIM
    sitk_pred_k = sitk.ReadImage(os.path.join(kidney_pred, pred))
    sitk_pred_array_k = sitk.GetArrayFromImage(sitk_pred_k)

    # PREDIÇÃO DA RECONSTRUÇÃO DA LESÃO QUE AGORA VOU CHAMAR DE SEGMENTAÇÃO DE LESÃO RUDIMENTAR
    sitk_pred_t_raw = sitk.ReadImage(os.path.join(tumor_raw_pred, pred))
    sitk_pred_array_t_raw = sitk.GetArrayFromImage(sitk_pred_t_raw)
    sitk_pred_array_t_raw = np.where(sitk_pred_array_t_raw == 3, 1, 0)

    pred_reconstructed = np.bitwise_or(
        sitk_pred_array_k, sitk_pred_array_t_raw)

    pred_reconstructed = post_processing_3d_two_biggest_elem(
        pred_reconstructed)

    # ETAPA 2  - Junção da interseção resultante com a segmentação refinada da lesão
    sitk_pred_t = sitk.ReadImage(os.path.join(tumor_pred, pred))
    sitk_pred_array_t = sitk.GetArrayFromImage(sitk_pred_t)
    sitk_pred_array_t = np.where(sitk_pred_array_t == 2, 1, 0)

    result = np.bitwise_or(pred_reconstructed, sitk_pred_array_t)

    result = sitk.GetImageFromArray(np.asarray(result, np.uint32))
    sitk.WriteImage(result, os.path.join(output, pred))
