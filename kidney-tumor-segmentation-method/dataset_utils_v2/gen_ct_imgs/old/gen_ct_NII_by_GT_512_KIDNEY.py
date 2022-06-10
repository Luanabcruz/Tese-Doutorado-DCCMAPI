'''
Script para gerar as imagens das lesões com base na marcação do rim pelo especialista
'''


import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import cv2
import glob

from methods.filters import *
from methods.delimitar_regioes_rins import *
from methods.center_images import *

from utils import create_nested_dir


def generate_imgs_masks_2d(path_root, output_dir, local, width, height, only_injuries_slices=False, histogram_specification=False):

    path_root = os.path.join(path_root, local)

    cont = 0
    folders = glob.glob(os.path.join(path_root, '*'))

    test_cases = [
        "case_00009",
        "case_00011",
        "case_00012",
        "case_00018",
        "case_00024",
        "case_00025",
        "case_00028",
        "case_00041",
        "case_00043",
        "case_00045",
        "case_00056",
        "case_00066",
        "case_00076",
        "case_00078",
        "case_00082",
        "case_00085",
        "case_00099",
        "case_00100",
        "case_00102",
        "case_00122",
        "case_00131",
        "case_00139",
        "case_00149",
        "case_00158",
        "case_00170",
        "case_00174",
        "case_00175",
        "case_00176",
        "case_00197",
        "case_00199",
        "case_00200",
    ]

    for folder in folders:

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        if not nome_paciente in test_cases:
            continue

        folder += "\\"

        print(nome_paciente)
        print("paciente: ", cont)
        cont += 1

        gt = sitk.ReadImage(folder+"segmentation.nii.gz")

        sitk.WriteImage(gt, os.path.join(
            output_dir, nome_paciente + ".nii.gz"))


if __name__ == '__main__':
    SPECIFICATION_TEMPLATE = "kidney_dataset_utils/imaging_176.nii.gz"

    width = 512
    height = 512

    dataset_type = ""

    # Dom
    dataset_root_dir = r"D:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master"
    # Lua
    # dataset_root_dir = r"D:\Documentos\Projetos\projeto-doutorado\base\base_img"

    output_dir = r'./GT_512_kits19'
    histogram_specification = False

    generate_imgs_masks_2d(dataset_root_dir, output_dir, dataset_type, width, height,
                           only_injuries_slices=False, histogram_specification=histogram_specification)
