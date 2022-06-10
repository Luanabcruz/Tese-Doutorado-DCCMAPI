

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

output = r'../pipeline_completo_nii/nii_pipeline_completo_lesao_final_lua'


for pred in os.listdir(kidney_pred):
    print(pred)

    '''
        ETAPA 1 - união da lesão rudimentar com a reconstrução
    '''

    # PREDIÇÃO DA LESÃO REFINADA
    sitk_pred_t = sitk.ReadImage(os.path.join(tumor_pred, pred))
    sitk_pred_array_t = sitk.GetArrayFromImage(sitk_pred_t)
    sitk_pred_array_t = np.where(sitk_pred_array_t == 2, 1, 0)

    # PREDIÇÃO DA RECONSTRUÇÃO DA LESÃO QUE AGORA VOU CHAMAR DE SEGMENTAÇÃO DE LESÃO RUDIMENTAR
    sitk_pred_t_raw = sitk.ReadImage(os.path.join(tumor_raw_pred, pred))
    sitk_pred_array_t_raw = sitk.GetArrayFromImage(sitk_pred_t_raw)
    sitk_pred_array_t_raw = np.where(sitk_pred_array_t_raw == 3, 1, 0)

    tumor_reconstructed = np.bitwise_or(
        sitk_pred_array_t, sitk_pred_array_t_raw)

    '''
        ETAPA 2 - união da saída da etapa 1 com o rim
    '''

    # PREDIÇÃO DO RIM
    sitk_pred_k = sitk.ReadImage(os.path.join(kidney_pred, pred))
    sitk_pred_array_k = sitk.GetArrayFromImage(sitk_pred_k)

    tumor_kidney_reconstructed = np.bitwise_or(
        tumor_reconstructed, sitk_pred_array_k)

    # aplicação do pós para capturar os dois maiores elementos
    tumor_kidney_reconstructed = post_processing_3d_two_biggest_elem(
        tumor_kidney_reconstructed)

    '''
        ETAPA 3 - interseção da saída da etapa 2 com a saída da etapa 1
    '''

    intersection = np.bitwise_and(
        tumor_kidney_reconstructed, tumor_reconstructed)

    # aplicação do pós da lesão.
    result = post_processing_3D(np.asarray(intersection, np.uint32))

    result = np.where(result == 1, 2, 0)

    result = sitk.GetImageFromArray(np.asarray(result, np.uint32))
    sitk.WriteImage(result, os.path.join(output, pred))
