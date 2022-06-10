
import sys
import torch
import numpy as np
import pandas as pd

from framework.lua_inference_2D import LuaInference2D
import framework.ct_dataset_manager as dm
import framework.ct_evaluator as ct_eval
from framework.ct_saver import CTSaver

from auth.save_df_to_sheets import save_in_lua_spreadsheet
from get_env import datasets_path

sys.path.append(r'E:\DoutoradoLuana\codes\luanet')

dataset ={
    "type": "cascade",
    "path":  r'D:\DoutoradoLuana\RESULTADOS_TESE\ensemble\kidney'
}

cases_info_dir = r"E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"

worksheet_name = "tumor_initial_ensembled"

folds = [i for i in range(11, 16)]
weights = []

for fold in folds:
    weights.append('../../PESOS_TESE/F{}/tumor/tumor_dpn131_f{}.pt'.format(fold,fold))

threshold = 0.5
batch_size = 12
cases = dm.load_dataset_dist(dataset_id="natal_v7")['test']

evaluator = ct_eval.CTEvaluator([
        'dice',
        'iou',
        'accuracy',
        'sensitivity',
        'specificity',
        'precision'
])

outdir = '../../RESULTADOS_TESE/ensemble/tumor_cascade_initial'

nii_saver =  CTSaver(outdir=outdir, inlabel= 1, outlabel= 2)

keys = ['case_name']
keys.extend(evaluator.get_metrics_name())

df = pd.DataFrame(columns=keys)
for case in cases:
    y_pred_summed = None
    y_true = None
    
    for weight in weights:
        model = torch.load(weight, map_location={'cuda:1': 'cuda:0'})
        
        post_proc = None

        inference = LuaInference2D(dataset['path'], model, evaluator, False, outdir, cases_info_dir=cases_info_dir)

        if dataset['type'] == 'cascade':
            inference.set_dataset_cascade()

        inference.set_dataset_dim_512()
        inference.set_nii_label(2)
        inference.set_channels(3)
        
        y_true, y_pred = inference.execute_one(case, batch_size)

        if y_pred_summed is None:
            y_pred_summed = y_pred
        else: 
            y_pred_summed = np.add(y_pred_summed, y_pred)    
    
    y_pred_summed = y_pred_summed/len(weights)

    y_pred_summed = np.where(y_pred_summed > threshold, 1, 0)

    nii_saver.save(case, y_pred_summed)

    result = evaluator.execute(y_true, y_pred_summed)
    
    result["case_name"] = case

    df = df.append(result, ignore_index=True)

    out_str = "{}, ".format(case) 
    for metric in evaluator.get_metrics_name():
        out_str += "{}, ".format(result[metric])

    print(out_str)

    EXCEL_ID = "1yi0Cnh_Q6acHBir60gsQFYdNrUUHApkugL4aEm9_jmc"
    
    try:
        save_in_lua_spreadsheet(worksheet_name, df,EXCEL_ID )  
    except Exception as e:
            print("Erro sao salvar no excel!")
            print(e)  

    