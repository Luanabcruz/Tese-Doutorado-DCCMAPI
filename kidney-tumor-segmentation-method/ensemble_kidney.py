
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

dataset = datasets_path()['kidney25D']

folds = [i for i in range(11, 16)]
weights = []

for fold in folds:
    weights.append('../../PESOS_TESE/F{}/kidney/kidney_resunet101_f{}.pt'.format(fold,fold))

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

outdir = '../../RESULTADOS_TESE/ensemble/kidney'

nii_saver =  CTSaver(outdir=outdir)

keys = ['case_name']
keys.extend(evaluator.get_metrics_name())

df = pd.DataFrame(columns=keys)
for case in cases:
    y_pred_summed = None
    y_true = None
    
    for weight in weights:
        model = torch.load(weight, map_location={'cuda:1': 'cuda:0'})
        
        inference = LuaInference2D(dataset, model, evaluator, False, './', cases_info_dir=None, prob_out = True)
        
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
    worksheet_name = "kidney_initial_ensembled"
    try:
        save_in_lua_spreadsheet(worksheet_name, df,EXCEL_ID )  
    except Exception as e:
            print("Erro sao salvar no excel!")
            print(e)  

    