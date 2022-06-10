from gspread.models import Worksheet
from auth.save_df_to_sheets import save_in_lua_spreadsheet
import torch
import framework.ct_evaluator as ct_eval
import framework.post_processings as pp
import framework.ct_dataset_manager as dm
from framework.lua_inference_2D import LuaInference2D
import sys
from kidney_dataset_utils.misc.split_kits19 import load_dpa_dist

def print_time(segundos):
    print("Total em segundos ", segundos)
    # dias = segundos // 86400
    segundos_rest = segundos % 86400
    horas = segundos_rest // 3600
    segundos_rest = segundos_rest % 3600
    minutos = segundos_rest // 60
    segundos_rest = segundos_rest % 60

    print("Tempo de execução de {} horas {} minutos e {} segundos".format(horas, minutos, segundos))

# Planilha da tese
EXCEL_ID = "1-3SdR6Ug4KAkrvt4VEK7hWyDsMS1j1NHH2wBqxB56Aw"

sys.path.append(r'E:\DoutoradoLuana\codes\luanet')

dataset = r'D:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_kidney_masks\Kits19_npz_25D_512_todas_fatias_rim'
save_nii = True
out_pred_folder_name = 'initial'
# folds = [i for i in range(4, 6)]
folds = [4]
for fold in folds:

    dpa_dist= 'DPA_group_HF{}.json'.format(fold)
    cases = load_dpa_dist(dpa_dist)['test']
    # cases = ['case_00009']
    worksheet_name = "method-kidney-HF{}-pos-2-fechamento-big".format(fold)
    weigths_path = '../../PESOS_TESE_KFOLD/HF{}/kidney/kidney_HF{}.pt'.format(fold,fold)
    outdir = r'D:\DoutoradoLuana\RESULTADOS_TESE_KFOLD\HF{}\kidney\{}'.format(fold, out_pred_folder_name)

    n_channels = 3
    model = None
    try:
        model = torch.load(weigths_path, map_location={'cuda:1': 'cuda:0'})
        model.cuda()
    except:
        print("Não foi possível carregar o modelo ", weigths_path)
        continue    
    

    evaluator = ct_eval.CTEvaluator([
            'dice',
            # 'iou',
            # 'accuracy',
            # 'sensitivity',
            # 'specificity',
            # 'precision'
    ])

    post_proc = None
    # post_proc = None

    inference = LuaInference2D(dataset, model, evaluator, save_nii, output_dir=outdir)
    inference.enable_verbose()
    inference.set_channels(n_channels)
    inference.set_nii_label(1)

    inference.execute(
        cases, 12, post_proc)

    segundos = inference.getLastExecution().sum()
    print_time(segundos)
        
    df = inference.get_result_dataframe()

    # try:
    #   save_in_lua_spreadsheet(worksheet_name, df,EXCEL_ID )  
    # except Exception as e:
    #   print("Erro sao salvar no excel!")
    #   print(e)  
