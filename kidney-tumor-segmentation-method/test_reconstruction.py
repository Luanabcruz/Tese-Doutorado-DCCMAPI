import torch
import framework.ct_evaluator as ct_eval
import framework.post_processings as pp
import framework.ct_dataset_manager as dm
from framework.lua_inference_2D import LuaInference2D
import sys
from auth.save_df_to_sheets import save_in_lua_spreadsheet
from kidney_dataset_utils.misc.split_kits19 import load_dpa_dist
sys.path.append(r'E:\DoutoradoLuana\codes\luanet')

def print_time(segundos):
    print("Total em segundos ", segundos)
    # dias = segundos // 86400
    segundos_rest = segundos % 86400
    horas = segundos_rest // 3600
    segundos_rest = segundos_rest % 3600
    minutos = segundos_rest // 60
    segundos_rest = segundos_rest % 60

    print("Tempo de execução de {} horas {} minutos e {} segundos".format(horas, minutos, segundos))

# Planilha da proposta de tese
EXCEL_ID = "1-3SdR6Ug4KAkrvt4VEK7hWyDsMS1j1NHH2wBqxB56Aw"


dataset = r'D:\DoutoradoLuana\datasets\gerados\Kits19_npz_25D_rebuild_tumor_512_sem_EH_test_all'
# Planilha da tese

# folds = [i for i in range(11, 16)]
folds = [3,5]
save_nii = True
for fold in folds:
    dpa_dist= 'DPA_group_HF{}.json'.format(fold)
    cases = load_dpa_dist(dpa_dist)['test']
    
    worksheet_name = "reconst_initial_HF{}".format(fold)
    weigths_path = '../../PESOS_TESE_KFOLD/HF{}/reconstruction/reconstruction_HF{}.pt'.format(fold,fold)
    # weigths_path = r'D:\DoutoradoLuana\codes\luanet_weights_sem_aug\unet_dice_resnet101_dist_dpa_runs_reconst_hf3_ch_3\weights_partial_diceval_epch25_20220121_10_26_11.pt'
    outdir = r'D:\DoutoradoLuana\RESULTADOS_TESE_KFOLD\HF{}\reconstruction-method-kidney\initial'.format(fold)

    model = torch.load(weigths_path, map_location={'cuda:1': 'cuda:0'})

    evaluator = ct_eval.CTEvaluator([
            'dice',
            # 'iou',
            # 'accuracy',
            # 'sensitivity',
            # 'specificity',
            # 'precision'
    ])

    # post_proc = [pp.min_slices_number_3D, pp.median_filter]
    post_proc = None

    inference = LuaInference2D(dataset, model, evaluator, save_nii, output_dir=outdir)
    inference.set_channels(3)
    #necessario por as imagens tem invervalo de 0 - 255 (erro meu na hora de gerar)
    inference.set_is_rgb(True)
    inference.set_nii_label(3)
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

import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)