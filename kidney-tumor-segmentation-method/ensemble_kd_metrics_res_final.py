
import os
import SimpleITK as sitk
import numpy as np
from sklearn.metrics import confusion_matrix
import framework.ct_dataset_manager as dm
import framework.ct_evaluator as ct_eval
import pandas as pd
from auth.save_df_to_sheets import save_in_lua_spreadsheet

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from kidney_utils.post_process import post_processing_3d_two_biggest_elem, post_processing_3D, post_processing_3D_morphological_closing

kits19_path =r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'

cases = dm.load_dataset_dist(dataset_id="natal_v7")['test']

evaluator = ct_eval.CTEvaluator([
                'dice',
                'iou',
                'accuracy',
                'sensitivity',
                'specificity',
                'precision'
])

keys = ['case_name']
keys.extend(evaluator.get_metrics_name())

pred_folder = r'D:\DoutoradoLuana\RESULTADOS_TESE\ENSEMBLE'
df = pd.DataFrame(columns=keys)

for case_name in cases:
    pred_name = case_name.replace("case_","prediction_")+".nii.gz"
    sitk_pred_k = sitk.ReadImage(os.path.join(pred_folder, 'kidney', 'final',pred_name))
    sitk_pred_array_k = sitk.GetArrayFromImage(sitk_pred_k)

    sitk_pred_array_k = np.where(sitk_pred_array_k == 1, 1, sitk_pred_array_k)
    sitk_pred_array_k = np.where(sitk_pred_array_k == 2, 1, sitk_pred_array_k)
    sitk_pred_array_k = sitk_pred_array_k.transpose(2, 0, 1)

    gt_k = sitk.ReadImage(os.path.join(
        kits19_path, case_name, "segmentation.nii.gz"))
    gt_array_k = sitk.GetArrayFromImage(gt_k)
    gt_array_k = np.where(gt_array_k == 1, 1, gt_array_k)
    gt_array_k = np.where(gt_array_k == 2, 1, gt_array_k)

    gt_array_k = gt_array_k.transpose(2, 0, 1)
    
    result = evaluator.execute(gt_array_k, sitk_pred_array_k, None)
    result["case_name"] = case_name

    df = df.append(result, ignore_index=True)

    out_str = "{}, ".format(case_name) 
    for metric in evaluator.get_metrics_name():
        out_str += "{}, ".format(result[metric])

    print(out_str)

    EXCEL_ID = "1yi0Cnh_Q6acHBir60gsQFYdNrUUHApkugL4aEm9_jmc"
    worksheet_name = "kidney_seg_final_ensemble"
    try:
        save_in_lua_spreadsheet(worksheet_name, df, EXCEL_ID )  
    except Exception as e:
            print("Erro sao salvar no excel!")
            print(e)  
