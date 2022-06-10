import os
from kidney_dataset_utils.misc.split_kits19 import load_dpa_dist
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    dataset_path = r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'

    dpa_dist = 'DPA_group_HF1.json'
    cases =  load_dpa_dist(dpa_dist)['test']
    sum_slices_ct = 0 
    sum_slices_kid = 0
    
    for case in tqdm(cases):
        case_path = os.path.join(dataset_path, case, "segmentation.nii.gz")
        sitk_case_k = sitk.ReadImage(case_path)
        sitk_case_array_k = sitk.GetArrayFromImage(sitk_case_k)
        
        sum_slices_ct += sitk_case_array_k.shape[2]

        ids = []
        sitk_case_array_k = sitk_case_array_k.transpose(2, 0, 1)
        for i in range(sitk_case_array_k.shape[0]):
            if(np.amax(sitk_case_array_k[i]) != 0):
                ids.append(i)

        new_gt = np.delete(sitk_case_array_k, ids, 0)       
        sum_slices_kid += new_gt.shape[0]
        
    print(sum_slices_ct)
    print(sum_slices_kid)