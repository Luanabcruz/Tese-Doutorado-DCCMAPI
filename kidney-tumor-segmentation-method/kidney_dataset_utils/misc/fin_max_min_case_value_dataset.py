import os
from tqdm import tqdm
from utils import create_nested_dir
import glob
import cv2
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

'''
    Encontra o maior e menor valor de valores da base
'''


def find_max_min(path_root):

    folders = glob.glob(os.path.join(path_root, '*'))
    max_value = None
    min_value = None

    for folder in tqdm(folders):

        nome_paciente = folder.replace(path_root, "").replace("\\", "")

        folder += "\\"

        tqdm.write(nome_paciente)

        imagem = sitk.ReadImage(folder+"imaging.nii.gz")
        imagem_array = sitk.GetArrayFromImage(imagem)

        if max_value is None:
            max_value = imagem_array.max()

        if min_value is None:
            min_value = imagem_array.min()

        if max_value < imagem_array.max():
            max_value = imagem_array.max()

        if min_value > imagem_array.min():
            min_value = imagem_array.min()

        tqdm.write("Max atual: "+str(max_value))
        tqdm.write("Min atual: "+str(min_value))

    print("\n\n\n")
    print("Max: ", str(max_value))
    print("Min: ", str(min_value))


if __name__ == '__main__':

    dataset_root_dir = r"D:\DOUTORADO_LUANA\bases2D\KiTS19_master"

    find_max_min(dataset_root_dir)
