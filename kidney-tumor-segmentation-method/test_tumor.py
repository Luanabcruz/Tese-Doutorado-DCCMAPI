from framework.metrics import jaccard
import torch
import framework.ct_evaluator as ct_eval
import framework.post_processings as pp
import framework.ct_dataset_manager as dm
from framework.lua_inference_2D import LuaInference2D
import sys
from auth.save_df_to_sheets import save_in_lua_spreadsheet
from kidney_dataset_utils.misc.split_kits19 import load_dpa_dist

sys.path.append(r'E:\DoutoradoLuana\codes\luanet')
# Planilha da tese
EXCEL_ID = "1yi0Cnh_Q6acHBir60gsQFYdNrUUHApkugL4aEm9_jmc"
dpa_dist = 'DPA_group_kf9.json'
# cases = dm.load_dataset_dist(dataset_id="natal_v7")['test']
cases =  load_dpa_dist(dpa_dist)['test']
# print(cases)
# exit()
dataset = r'D:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_tumor_masks\KiTS19_master_npz'

# worksheet_name = "tumor256_all_metrics_f{}".format(fold)
weigths_path = r'C:\Users\domin\Desktop\kf8\weights_final.pt'

outdir = r'D:\DoutoradoLuana\RESULTADOS_TESE\F11\tumor\t256'
save_nii = False
model = None

try:
        model = torch.load(weigths_path, map_location={'cuda:1': 'cuda:0'})
        model.cuda()
except:
        print("Não foi possível carregar o modelo ", weigths_path)
        exit() 


evaluator = ct_eval.CTEvaluator([
        'dice',
        # 'iou',
        # 'accuracy',
        # 'sensitivity',
        # 'specificity',
        # 'precision'
])


# post_proc = [pp.min_slices_number_3D]
post_proc = None

inference = LuaInference2D(dataset, model, evaluator, save_nii, outdir)
inference.set_channels(3)
inference.set_nii_label(2)
inference.execute(
cases, 12, post_proc)

df = inference.get_result_dataframe()

# try:
#         save_in_lua_spreadsheet(worksheet_name, df,EXCEL_ID )  
# except Exception as e:
#         print("Erro sao salvar no excel!")
#         print(e)  
