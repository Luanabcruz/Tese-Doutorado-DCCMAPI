import torch
import framework.ct_evaluator as ct_eval
import framework.post_processings as pp
import framework.ct_dataset_manager as dm
from framework.lua_inference_2D import LuaInference2D
import sys
from auth.save_df_to_sheets import save_in_lua_spreadsheet

sys.path.append(r'E:\DoutoradoLuana\codes\luanet')

# Planilha da proposta de tese
EXCEL_ID = "1yi0Cnh_Q6acHBir60gsQFYdNrUUHApkugL4aEm9_jmc"


dataset = r'D:\DoutoradoLuana\datasets\gerados\Kits19_npz_25D_rebuild_tumor_512_sem_EH_test'
# Planilha da tese

cases = dm.load_dataset_dist(dataset_id="natal_v7")['test']


folds = [13]

for fold in folds:
    worksheet_name = "reconst_all_metrics_f{}_v2".format(fold)
    weigths_path = '../../PESOS_TESE/F{}/reconstruction/reconstruction_resunet101_f{}_v2.pt'.format(fold,fold)
    outdir = r'D:\DoutoradoLuana\RESULTADOS_TESE\F{}\reconstruction_v2'.format(fold)

    model = torch.load(weigths_path, map_location={'cuda:1': 'cuda:0'})

    evaluator = ct_eval.CTEvaluator([
            'dice',
            'iou',
            'accuracy',
            'sensitivity',
            'specificity'
    ])

    # post_proc = [pp.min_slices_number_3D, pp.median_filter]
    post_proc = None

    inference = LuaInference2D(dataset, model, evaluator, False, output_dir=outdir)
    inference.set_channels(3)
    #necessario por as imagens tem invervalo de 0 - 255 (erro meu na hora de gerar)
    inference.set_is_rgb(True)
    inference.set_nii_label(3)
    inference.execute(
        cases, 12, post_proc)

    df = inference.get_result_dataframe()

    try:
      save_in_lua_spreadsheet(worksheet_name, df,EXCEL_ID )  
    except Exception as e:
      print("Erro sao salvar no excel!")
      print(e)  

