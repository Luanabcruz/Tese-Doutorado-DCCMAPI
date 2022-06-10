import numpy as np
import SimpleITK as sitk
import glob

from methods.filters import *
from methods.delimitar_regioes_rins import *
from methods.center_images import *

import utils_ds as utl
from tqdm import tqdm

import os

from preprocessings import preprocessing_spec_hist


def generate_npz(in_path, out_dir, shape, preprocess=None):
    width, height = shape
    folders = glob.glob(os.path.join(in_path, '*'))

    for folder in tqdm(folders):

        case_name = folder.replace(in_path, "").replace("\\", "")

        folder += "\\"

        tqdm.write(case_name)

        # carrega sitk e array
        image_array, _ = utl.load_sitk(folder+"imaging.nii.gz")

        if preprocess is not None:
            image_array = preprocess(image_array)
        # janelamento
        image_array = windowing(image_array)
        # muda os valores para 0 e 1
        image_array = rescale_intensity(image_array)

        utl.save_image_npz(image_array, None, case_name, out_dir)


if __name__ == '__main__':
    '''
    ESSE SCRIPT É PARA GERAR AS TEXTURAS DO CHALLENGE, JÁ QUE NÃO POSSUI AS MÁSCARAS dO ESPECIALISTA
    '''

    in_dir = r"D:\DOUTORADO_LUANA\etapa1\bases2D\teste challenge\teste challenge"
    out_dir = r'E:\DoutoradoLuana\datasets\gerados\challenge\kits19_no_GT'

    # preprocess = preprocessing_spec_hist
    preprocess = None
    shape = (512, 512)
    params = {
        "in_dir": in_dir,
        "out_dir": out_dir,
        "preprocess": str(preprocess),
    }

    utl.create_nested_dir(out_dir)
    utl.save_params(out_dir, params)

    generate_npz(in_dir, out_dir, shape, preprocess)
