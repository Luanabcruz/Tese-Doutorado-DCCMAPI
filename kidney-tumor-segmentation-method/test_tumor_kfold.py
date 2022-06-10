from framework.metrics import jaccard
import torch
import framework.ct_evaluator as ct_eval
import framework.post_processings as pp
import framework.ct_dataset_manager as dm
from framework.lua_inference_2D import LuaInference2D
import sys
from auth.save_df_to_sheets import save_in_lua_spreadsheet
from kidney_dataset_utils.misc.split_kits19 import load_dpa_dist

EXCEL_ID = "1-3SdR6Ug4KAkrvt4VEK7hWyDsMS1j1NHH2wBqxB56Aw"

sys.path.append(r'E:\DoutoradoLuana\codes\luanet')

dataset = r'D:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_tumor_masks\KiTS19_master_npz'
save_nii = False

for fold in range(1,6):

        weigths_path = r'D:\DoutoradoLuana\PESOS_TESE_KFOLD\HF{}\tumor\tumor_HF{}.pt'.format(fold,fold)
        outdir = r'D:\DoutoradoLuana\RESULTADOS_TESE\METHOD_TUMOR\HF{}\GT-kid-tumor'.format(fold)
        
        worksheet_name = "method-tumor-256-hf{}-pos-fcfm".format(fold)

        dpa_dist = 'DPA_group_HF{}.json'.format(fold)
        cases =  load_dpa_dist(dpa_dist)['test']

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

        # post_proc = [pp.min_slices_number_3D, pp.morphological_closing_3D ]
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
