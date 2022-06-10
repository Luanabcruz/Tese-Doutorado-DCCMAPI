

import sys
import numpy as np
import SimpleITK as sitk
import os
import pathlib

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from kidney_utils.post_process import post_processing_3d_two_biggest_elem, post_processing_3D

tumor_pred = r'C:\Users\usuario\Desktop\saida_lesao_reconstruida'  # reconstrucao

output = r'C:\Users\usuario\Desktop\saida'


for pred in os.listdir(tumor_pred):
    print(pred)

    sitk_pred_t = sitk.ReadImage(os.path.join(tumor_pred, pred))
    sitk_pred_array_t = sitk.GetArrayFromImage(sitk_pred_t)
    sitk_pred_array_t = np.where(sitk_pred_array_t == 2, 1, 0)

    sitk_pred_array_t = post_processing_3d_two_biggest_elem(
        np.asarray(sitk_pred_array_t, np.uint32))

    sitk_pred_array_t = np.where(sitk_pred_array_t == 1, 2, 0)

    sitk_pred_array_t = sitk.GetImageFromArray(
        np.asarray(sitk_pred_array_t, np.uint32))
    sitk.WriteImage(sitk_pred_array_t, os.path.join(output, pred))
