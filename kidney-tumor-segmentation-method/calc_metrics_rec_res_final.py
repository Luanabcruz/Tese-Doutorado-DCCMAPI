
import os
import SimpleITK as sitk
import numpy as np
from sklearn.metrics import confusion_matrix
import framework.ct_dataset_manager as dm
import framework.ct_evaluator as ct_eval
import pandas as pd
from auth.save_df_to_sheets import save_in_lua_spreadsheet
from kidney_dataset_utils.misc.split_kits19 import load_dpa_dist

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from kidney_utils.post_process import post_processing_3d_two_biggest_elem, post_processing_3D, post_processing_3D_morphological_closing

kits19_path =r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'

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
folds = [i for i in range(11, 15)]
folds = [5]
for fold in folds:
    pred_folder = r'D:\DoutoradoLuana\RESULTADOS_TESE_KFOLD\HF{}'.format(fold)
    df = pd.DataFrame(columns=keys)

    dpa_dist= 'DPA_group_HF{}.json'.format(fold)
    cases = load_dpa_dist(dpa_dist)['test']

    for case_name in cases:
        pred_name = case_name.replace("case_","prediction_")+".nii.gz"
        sitk_pred_k = sitk.ReadImage(os.path.join(pred_folder, 'reconstruction', 'final',pred_name))
        sitk_pred_array_k = sitk.GetArrayFromImage(sitk_pred_k)

        sitk_pred_array_k = np.where(sitk_pred_array_k == 1, 0, sitk_pred_array_k)
        sitk_pred_array_k = np.where(sitk_pred_array_k == 2, 1, sitk_pred_array_k)
        sitk_pred_array_k = np.where(sitk_pred_array_k == 3, 1, sitk_pred_array_k)
        sitk_pred_array_k = sitk_pred_array_k.transpose(2, 0, 1)

        gt_k = sitk.ReadImage(os.path.join(
            kits19_path, case_name, "segmentation.nii.gz"))
        gt_array_k = sitk.GetArrayFromImage(gt_k)
        gt_array_k = np.where(gt_array_k == 1, 0, gt_array_k)
        gt_array_k = np.where(gt_array_k == 2, 1, gt_array_k)

        gt_array_k = gt_array_k.transpose(2, 0, 1)
        
        result = evaluator.execute(gt_array_k, sitk_pred_array_k, None)
        result["case_name"] = case_name

        df = df.append(result, ignore_index=True)

        out_str = "{}, ".format(case_name) 
        for metric in evaluator.get_metrics_name():
            out_str += "{}, ".format(result[metric])

        print(out_str)

        EXCEL_ID = "1-3SdR6Ug4KAkrvt4VEK7hWyDsMS1j1NHH2wBqxB56Aw"
        worksheet_name = "cascade-reconst_final_HF{}".format(fold)
        try:
            save_in_lua_spreadsheet(worksheet_name, df,EXCEL_ID )  
        except Exception as e:
                print("Erro sao salvar no excel!")
                print(e)  
