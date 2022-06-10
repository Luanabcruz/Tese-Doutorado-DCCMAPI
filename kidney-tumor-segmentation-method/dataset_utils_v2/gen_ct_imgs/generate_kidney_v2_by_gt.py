

import matplotlib.pyplot as plt
import numpy as np

import glob

from methods.filters import *
from methods.delimitar_regioes_rins import *
from methods.center_images import *

import utils_ds as utl

from preprocessings import preprocessing_spec_hist

from tqdm import tqdm
import os


def generate_npz(in_path, out_dir, only_kidney, preprocess=None):

    folders = glob.glob(os.path.join(in_path, '*'))

    # test_cases = ["case_00009", "case_00011", "case_00012", "case_00018", "case_00024", "case_00025", "case_00028", "case_00041", "case_00043", "case_00045", "case_00056", "case_00066", "case_00076", "case_00078", "case_00082",
    #               "case_00085", "case_00099", "case_00100", "case_00102", "case_00122", "case_00131", "case_00139", "case_00149", "case_00158", "case_00170", "case_00174", "case_00175", "case_00176", "case_00197", "case_00199", "case_00200"]

    for folder in tqdm(folders):

        case_name = folder.replace(in_path, "").replace("\\", "")

        # if not case_name in test_cases:
        #     continue

        folder += "\\"

        tqdm.write(case_name)

        # carrega sitk e array
        image_array, image = utl.load_sitk(folder+"imaging.nii.gz")
        gt_array, _ = utl.load_sitk(folder+"segmentation.nii.gz")

        if preprocess is not None:
            image_array = preprocess(image_array)
        # janelamento
        image_array = windowing(image_array)
        # muda os valores para 0 e 1
        image_array = rescale_intensity(image_array, rgb_scale=True)

        if only_kidney:
            image_array, gt_array = remove_empty_slices_sem_rins(
                image_array, gt_array)

        # tumor label becomes kidney label
        # gt_array = np.array(np.where(gt_array == 2, 1, gt_array), np.uint8)

        # salvar apenas as lesões.
        gt_array = np.array(np.where(gt_array == 1, 0, gt_array), np.uint8)
        gt_array = np.array(np.where(gt_array == 2, 1, gt_array), np.uint8)

        utl.save_image_npz(image_array, gt_array, case_name, out_dir)


if __name__ == '__main__':

    in_dir = r"E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"
    out_dir = r'D:\DoutoradoLuana\datasets\gerados\Kits19_npz_25D_rebuild_tumor_512_sem_EH_test_all'

    # preprocess = preprocessing_spec_hist
    only_kidney = False
    preprocess = None

    params = {
        "in_dir": in_dir,
        "out_dir": out_dir,
        "preprocess": str(preprocess),
        "only_kidney": only_kidney
    }

    utl.create_nested_dir(out_dir)
    utl.save_params(out_dir, params)

    generate_npz(in_dir, out_dir, only_kidney, preprocess)
