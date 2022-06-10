from framework.metrics import jaccard
import torch
import framework.ct_evaluator as ct_eval
import framework.post_processings as pp
import framework.ct_dataset_manager as dm
from framework.lua_inference_2D import LuaInference2D
import sys
from kidney_dataset_utils.misc.split_kits19 import load_dpa_dist
from auth.save_df_to_sheets import save_in_lua_spreadsheet

# Planilha da tese
EXCEL_ID = "1-3SdR6Ug4KAkrvt4VEK7hWyDsMS1j1NHH2wBqxB56Aw"

sys.path.append(r'E:\DoutoradoLuana\codes\luanet')
cases_info_dir = r"E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"

# dataset ={
# "type": "image",
# "path": r"C:\Users\domin\Desktop\out_kidney_test\pred_image"
# }

folds = [i for i in range(1,6)]
folds = [4]
evaluator = ct_eval.CTEvaluator([
        'dice',
        'iou',
        'accuracy',
        'sensitivity',
        'specificity',
        'precision'
])


for fold in folds:
        dpa_dist= 'DPA_group_HF{}.json'.format(fold)
        cases = load_dpa_dist(dpa_dist)['test']

        weigths_path = '../../PESOS_TESE_KFOLD/HF{}/tumor/tumor_HF{}.pt'.format(fold,fold)
        outdir = r'D:\DoutoradoLuana\RESULTADOS_TESE_KFOLD\HF{}\tumor\refinamento_anselmo'.format(fold)

        dataset ={
        "type": "cascade",
        "path":  r'D:\DoutoradoLuana\RESULTADOS_TESE_KFOLD\HF{}\kidney\method-kid\rec_pos'.format(fold)
        }

        worksheet_name = "no-holly-refinamento_anselmo_HF{}".format(fold)

        model = None
        try:
                model = torch.load(weigths_path, map_location={'cuda:1': 'cuda:0'})
                model.cuda()
        except Exception as e:
                print("Não foi possível carregar o modelo ", weigths_path)
                print(e)
                continue  

        post_proc = [pp.min_slices_number_3D]
        # post_proc = None

        inference = LuaInference2D(dataset['path'], model, evaluator, False, outdir, cases_info_dir=cases_info_dir)

        if dataset['type'] == 'cascade':
                inference.set_dataset_cascade()

        inference.set_dataset_dim_512()

        inference.set_channels(3)
        inference.set_nii_label(2)
        inference.execute(
        cases, 12, post_proc)

        df = inference.get_result_dataframe()

        try:
                save_in_lua_spreadsheet(worksheet_name, df,EXCEL_ID )  
        except Exception as e:
                print("Erro sao salvar no excel!")
                print(e)