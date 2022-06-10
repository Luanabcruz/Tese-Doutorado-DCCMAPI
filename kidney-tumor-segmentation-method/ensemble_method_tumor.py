
import sys
from SimpleITK.SimpleITK import Threshold
import torch
from framework.lua_inference_2D import LuaInference2D
import framework.ct_dataset_manager as dm
import numpy as np
import framework.ct_evaluator as ct_eval
sys.path.append(r'E:\DoutoradoLuana\codes\luanet')

dataset = r'D:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_tumor_masks\KiTS19_master_npz'

weights = [
    '../../PESOS_TESE/F13/tumor/tumor_dpn131_f11.pt',
    '../../PESOS_TESE/F13/tumor/tumor_dpn131_f12.pt',
    '../../PESOS_TESE/F13/tumor/tumor_dpn131_f13.pt',
    '../../PESOS_TESE/F13/tumor/tumor_dpn131_f14.pt',
    '../../PESOS_TESE/F13/tumor/tumor_dpn131_f15.pt'
]

threshold = 0.5
batch_size = 12
cases = dm.load_dataset_dist(dataset_id="natal_v7")['test']

evaluator = ct_eval.CTEvaluator([
        'dice'
])

for case in cases:
    y_pred_summed = None
    y_true = None
    
    for weight in weights:
        model = torch.load(weight)
        
        inference = LuaInference2D(dataset, model, evaluator, False, './', cases_info_dir=None)    
        y_true, y_pred = inference.execute_one(case, batch_size)

        if y_pred_summed is None:
            y_pred_summed = y_pred
        else: 
            y_pred_summed = np.add(y_pred_summed, y_pred)    
    
    y_pred_summed = y_pred_summed/len(weights)

    y_pred_summed = np.where(y_pred_summed > 0.5, 1, 0)
    metric_val = evaluator.execute(y_true, y_pred_summed)
    print("{}: {}".format(case, metric_val['dice']))
    