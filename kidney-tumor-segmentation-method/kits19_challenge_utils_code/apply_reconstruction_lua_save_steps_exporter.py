

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
# tumor_raw_pred = r'../pipeline_completo_nii/nii_pred_resunet101_reconstrucao_lesao'
tumor_raw_pred = r'../nii_luaExporter/reconstrucao/inicial'

tumor_pred = r'../pipeline_completo_nii/nii_pred_dpn131_lesao_resized_512'  # reconstrucao

output = r'../nii_luaExporter'

kits19_path = r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'


def save_nii_step(image_array, prefix, nii_file_info=None):
    image = sitk.GetImageFromArray(np.asarray(image_array, np.uint32))
    if not os.path.isdir(os.path.join(output, prefix)):
        os.makedirs(os.path.join(output, prefix))

    if nii_file_info is not None:
        image.CopyInformation(nii_file_info)

    sitk.WriteImage(image, os.path.join(output, prefix, pred))


# for pred in ['prediction_00174.nii.gz']:
for pred in os.listdir(tumor_raw_pred):
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
        ETAPA 1 - união da lesão rudimentar com a reconstrução
    '''

    # PREDIÇÃO DA LESÃO REFINADA (a nomenclatura de Lua é "segmentação inicial lesão")
    sitk_pred_t = sitk.ReadImage(os.path.join(tumor_pred, pred))
    sitk_pred_array_t = sitk.GetArrayFromImage(sitk_pred_t)
    # salvar segmentação inicial (lua nomenclatura)
    # save_nii_step(sitk_pred_array_t, os.path.join('segmentacao_inicial','lesao'), im_k)
    
    sitk_pred_array_t = np.where(sitk_pred_array_t == 2, 1, 0)

    

    # PREDIÇÃO DA RECONSTRUÇÃO DA LESÃO QUE AGORA VOU CHAMAR DE SEGMENTAÇÃO DE LESÃO RUDIMENTAR
    sitk_pred_t_raw = sitk.ReadImage(os.path.join(tumor_raw_pred, pred))
    sitk_pred_array_t_raw = sitk.GetArrayFromImage(sitk_pred_t_raw)
    sitk_pred_array_t_raw = np.where(sitk_pred_array_t_raw == 3, 1, 0)
    
    # Salvar o que Lua chama de reconstrução inicial
    # mudando para 2 para salvar.. é importante para o exporter
    aux =  np.where(sitk_pred_array_t_raw.copy() == 1, 2, 0)
    # save_nii_step(aux, os.path.join('reconstrucao', 'inicial'), im_k)
    aux = None

    
    tumor_reconstructed = np.bitwise_or(
        sitk_pred_array_t, sitk_pred_array_t_raw)


    # Salvar o que Lua chama de reconstrução final
    # mudando para 2 para salvar.. é importante para o exporter
    aux =  np.where(tumor_reconstructed.copy() == 1, 2, 0)
    # save_nii_step(aux, os.path.join('reconstrucao', 'final'), im_k)
    aux = None
    # continue
    
    '''
        ETAPA 2 - união da saída da etapa 1 com o rim
    '''

    # PREDIÇÃO DO RIM
    sitk_pred_k = sitk.ReadImage(os.path.join(kidney_pred, pred))
    sitk_pred_array_k = sitk.GetArrayFromImage(sitk_pred_k)
    # salvar segmentação inicial do rim (lua nomenclatura)
    # save_nii_step(sitk_pred_array_k, os.path.join('segmentacao_inicial','rim'), im_k)
    
    
    tumor_kidney_reconstructed = np.bitwise_or(
        tumor_reconstructed, sitk_pred_array_k)

    # salvar segmentação final - apenas união (lua nomenclatura)
    # save_nii_step(tumor_kidney_reconstructed, os.path.join('segmentacao_final','rim','uniao'), im_k)

    # aplicação do pós para capturar os dois maiores elementos
    tumor_kidney_reconstructed = post_processing_3d_two_biggest_elem(
        tumor_kidney_reconstructed)

    # salvar segmentação final -  união seguido com pós dos maiores elementos (lua nomenclatura)
    # save_nii_step(tumor_kidney_reconstructed, os.path.join('segmentacao_final','rim','maiores_elementos'), im_k)

    # exit()
    '''
        ETAPA 3 - interseção da saída da etapa 2 com a saída da etapa 1
    '''

    intersection = np.bitwise_and(
        tumor_kidney_reconstructed, tumor_reconstructed)

   

    
    intersection  = np.where(intersection == 1, 2, 0)
        
    # salvar segmentação final -  apenas interseção (lua nomenclatura)
    # save_nii_step(intersection, os.path.join('segmentacao_final','lesao','intersecao'), im_k)
    
    # aplicação do pós da lesão.
    result = post_processing_3D(np.asarray(intersection.copy(), np.uint32))

    # result = np.where(result == 1, 2, 0)
    # salvar segmentação final -  interseção seguida de pós (lua nomenclatura)
    save_nii_step(result, os.path.join('segmentacao_final','lesao','descontinuosv2'), im_k)

    # result = sitk.GetImageFromArray(np.asarray(result, np.uint32))
    # sitk.WriteImage(result, os.path.join(output, pred))
