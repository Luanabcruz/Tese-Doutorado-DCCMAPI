import os 
import numpy as np
import framework.ct_dataset_manager as dm

import SimpleITK as sitk

dataset = r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'

cases = dm.load_dataset_dist(dataset_id="natal_v7")['test']

len_all_slices = 0
len_all_pixels = 0
num_cases = len(cases)
cases = ["case_00176"]
for case in cases:
    image_path = os.path.join(dataset, case,'segmentation.nii.gz')
    imagem = sitk.ReadImage(image_path)
    imagem_array = sitk.GetArrayFromImage(imagem)
    imagem_array = imagem_array.transpose(2, 0, 1)
    ids = []
    for i in range(imagem_array.shape[0]):
        if(np.amax(imagem_array[i]) != 0):
            ids.append(i)
    new_gt = np.delete(imagem_array, ids, 0)

    # print(new_gt.shape)
    # exit()
    len_all_slices += new_gt.shape[0]
  

print("Num cases: ", num_cases)   
print("Num de fatias: ", len_all_slices)