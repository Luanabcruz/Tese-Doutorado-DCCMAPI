import warnings
warnings.filterwarnings("ignore")

from framework.metrics import jaccard
import torch
import framework.ct_evaluator as ct_eval
import framework.post_processings as pp
import framework.ct_dataset_manager as dm
from framework.lua_inference_2D import LuaInference2D
import sys
from tqdm import tqdm

from paper_exec_params import get_params
from auth.save_df_to_sheets import save_in_lua_spreadsheet

sys.path.append(r'D:\DoutoradoLuana\codes\pytorch-deeplab-xception')

CASES_INFO_DIR=r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'
DATASET_PATH_ROOT = r'D:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_tumor_masks\KiTS19_master_npz'
CASES_TEST = dm.load_dataset_dist(dataset_id="natal_v7")['test']



experiments = get_params()

for params in tqdm(experiments):
        
        tqdm.write("Experimento: {}".format(params['name']))

        model = torch.load(params['weights'])


        evaluator = ct_eval.CTEvaluator([
                'dice',
                'iou',
                'accuracy',
                'sensitivity',
                'specificity',
                'precision',
                'hausdorff',
                'average_surface_distance'
        ])

        if params['postprocessing']:
                post_proc = [pp.min_slices_number_3D]
        else:
                post_proc = None

        out_foldername = "{}_{}".format(params['name'], "COM_POS" if params['preprocessing'] else "SEM_POS")
        output_dir = "./out/tumor/paper_revisions_results/{}".format(out_foldername)
        inference = LuaInference2D(
                DATASET_PATH_ROOT, 
                model, 
                evaluator, 
                False,
                output_dir=output_dir,
                cases_info_dir=CASES_INFO_DIR
        )

        inference.disable_verbose()
        inference.set_channels(3)
        inference.execute(CASES_TEST, False, post_proc)

        df = inference.get_result_dataframe()
        
        try:
                save_in_lua_spreadsheet(out_foldername,df)
        except Exception as e:
                print("Erro sao salvar no excel!")
                print(e)
        # break