from kidney_utils.dataset import load_casebyname_from_image_folder
import os
import SimpleITK as sitk
import numpy as np

cases = ["case_00006", "case_00012", "case_00021", "case_00022", "case_00029", "case_00034", "case_00056", "case_00057", "case_00070", "case_00073", "case_00074", "case_00078", "case_00080", "case_00082", "case_00083",
         "case_00085", "case_00087", "case_00095", "case_00105", "case_00108", "case_00117", "case_00130", "case_00134", "case_00140", "case_00142", "case_00143", "case_00150", "case_00153", "case_00160", "case_00181", "case_00184"]

dataset = r'C:\Users\usuario\Documents\UFMA\DoutoradoLuana\segmentacao2d\luanet-segmentation-2d\KiTS19_master_imagens_512_orig'
output_dir = 'output_gt_nii'
for case_name in cases:

    case = load_casebyname_from_image_folder(
        dataset, case_name, image_folder='Images', mask_folder='Masks')

    gt_array = case['mask']

    gt_array = np.where(gt_array == 255, 1, 0)

    gt = sitk.GetImageFromArray(gt_array)
    sitk.WriteImage(gt, os.path.join(
        output_dir, 'gt_{}.nii.gz'.format(case_name)))
