from gspread.models import Worksheet
from auth.save_df_to_sheets import save_in_lua_spreadsheet
import torch
import framework.ct_evaluator as ct_eval
import framework.post_processings as pp
import framework.ct_dataset_manager as dm
from framework.lua_inference_2D import LuaInference2D
import sys
from kidney_dataset_utils.misc.split_kits19 import load_dpa_dist

# Planilha da tese
EXCEL_ID = "1-3SdR6Ug4KAkrvt4VEK7hWyDsMS1j1NHH2wBqxB56Aw"

sys.path.append(r'E:\DoutoradoLuana\codes\luanet')
fold = 5
dpa_dist = 'DPA_group_HF{}.json'.format(fold)
cases =  load_dpa_dist(dpa_dist)['test']

dataset = r'D:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_kidney_masks\Kits19_npz_25D_512_todas_fatias_rim'
save_nii = False

worksheet_name = "method-kidney-hf{}".format(fold)
weigths_path = r'C:\Users\domin\Desktop\kidney-kfold\ultimos\hf{}\kidney_hf{}.pt'.format(fold, fold)
outdir = r'C:\Users\domin\Desktop\kidney-kfold'

n_channels = 3
model = None

try:
    model = torch.load(weigths_path, map_location={'cuda:1': 'cuda:0'})
    model.cuda()
except:
    print("Não foi possível carregar o modelo ", weigths_path)
    exit()


evaluator = ct_eval.CTEvaluator([
        'dice',
        'iou',
        'accuracy',
        'sensitivity',
        'specificity',
        'precision'
])

# post_proc = [pp.two_biggest_elem]
post_proc = None

inference = LuaInference2D(dataset, model, evaluator, save_nii, output_dir=outdir)
inference.enable_verbose()
inference.set_channels(n_channels)
inference.set_nii_label(1)
inference.execute(
    cases, 12, post_proc)

df = inference.get_result_dataframe()

try:
    save_in_lua_spreadsheet(worksheet_name, df,EXCEL_ID )  
except Exception as e:
    print("Erro sao salvar no excel!")
    print(e)  
