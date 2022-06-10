

import sys
import numpy as np
import SimpleITK as sitk
import os

from kidney_utils.post_process import post_processing_3d_two_biggest_elem, post_processing_3D
from get_env import datasets_path

kits19_path = datasets_path()['kits19']

folds = [i for i in range(1, 6)]

folds = [4]

for fold in folds:
    print("Fold-{}\n".format(fold))
    kidney_pred = r'D:\DoutoradoLuana\RESULTADOS_TESE_KFOLD\HF{}\kidney\initial'.format(fold)
    reconstruction_pred = r'D:\DoutoradoLuana\RESULTADOS_TESE_KFOLD\HF{}\reconstruction-method-kidney\initial'.format(fold)
   
    output = r'D:\DoutoradoLuana\RESULTADOS_TESE_KFOLD\HF{}'.format(fold)

    def save_nii_step(image_array, prefix, nii_file_info=None):
        image = sitk.GetImageFromArray(np.asarray(image_array, np.uint32))
        if not os.path.isdir(os.path.join(output, prefix)):
            os.makedirs(os.path.join(output, prefix))

        if nii_file_info is not None:
            image.CopyInformation(nii_file_info)

        sitk.WriteImage(image, os.path.join(output, prefix, pred))

    for pred in os.listdir(reconstruction_pred):
        if os.path.splitext(pred)[-1] != '.gz':
            continue
        
        print(pred)
        print("\n")
        '''
        Carregar GT
        '''
        case_name = "case_"+pred.replace("prediction_", "").replace(".nii.gz", "")

        im_k = sitk.ReadImage(os.path.join(
            kits19_path, case_name, "imaging.nii.gz"))

        gt_k = sitk.ReadImage(os.path.join(
            kits19_path, case_name, "segmentation.nii.gz"))

        gt_array_k = sitk.GetArrayFromImage(gt_k)

        gt_array_k = np.where(gt_array_k == 1, 0, gt_array_k)
        gt_array_k = np.where(gt_array_k == 2, 1, gt_array_k)

        gt_array_k = sitk.GetArrayFromImage(gt_k)
        gt_array_k = np.where(gt_array_k == 2, 1, gt_array_k)

        '''
            ETAPA 1 - união da saída da reconstrução da lesão com o rim
        '''

        # PREDIÇÃO DA RECONSTRUÇÃO DA LESÃO QUE AGORA VOU CHAMAR DE SEGMENTAÇÃO DE LESÃO RUDIMENTAR
        sitk_pred_t_raw = sitk.ReadImage(os.path.join(reconstruction_pred, pred))
        sitk_pred_array_t_raw = sitk.GetArrayFromImage(sitk_pred_t_raw)
        sitk_pred_array_t_raw = np.where(sitk_pred_array_t_raw == 3, 1, 0)
                
       

        # PREDIÇÃO DO RIM
        sitk_pred_k = sitk.ReadImage(os.path.join(kidney_pred, pred))
        sitk_pred_array_k = sitk.GetArrayFromImage(sitk_pred_k)
        # salvar segmentação inicial do rim (lua nomenclatura)
        # save_nii_step(sitk_pred_array_k, os.path.join('segmentacao_inicial','rim'), im_k)
        
        tumor_kidney_reconstructed = np.bitwise_or(
            sitk_pred_array_t_raw, sitk_pred_array_k)

        # salvar segmentação final - apenas união (lua nomenclatura)
        save_nii_step(tumor_kidney_reconstructed, os.path.join('kidney','method-kid','rec_initial'), im_k)

        '''
            ETAPA 1 - aplica o pós
        '''

        # aplicação do pós para capturar os dois maiores elementos
        tumor_kidney_reconstructed = post_processing_3d_two_biggest_elem(
            tumor_kidney_reconstructed)

        # salvar segmentação final -  união seguido com pós dos maiores elementos (lua nomenclatura)
        save_nii_step(tumor_kidney_reconstructed, os.path.join('kidney','method-kid','rec_pos'), im_k)

        

import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)