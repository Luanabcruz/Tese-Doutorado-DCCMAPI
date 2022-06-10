import os
import numpy as np

import SimpleITK as sitk


def create_nested_dir(path, destroy_dir=True):
    # Verifica a existência do diretório, para criá-lo caso não exista.
    if not os.path.exists(path):
        os.makedirs(path)


def post_processing_3D(gt_array):

    gt_array_rins = np.zeros(gt_array.shape)
    gt_array_rins[np.where(gt_array != 0)] = 1
    gt_array_rins = gt_array_rins.astype(np.uint8)

    gt_rins = sitk.GetImageFromArray(gt_array_rins)
    gt_transpose = sitk.GetImageFromArray(gt_array)  # classes originais

    stats = sitk.LabelIntensityStatisticsImageFilter()
    cc_intensity = sitk.ConnectedComponent(gt_rins)
    stats.Execute(cc_intensity, gt_rins)

    gt_original_array = np.zeros(gt_array.shape)
    volume_original_array = np.zeros(gt_array.shape)

    for ps_label in stats.GetLabels():

        # if(ps_label == None):
        # 	continue

        array_cc = sitk.GetArrayFromImage(cc_intensity)
        array_cc = ((array_cc == ps_label)*1)

        array_cc = sitk.GetImageFromArray(array_cc)

        stats = sitk.LabelStatisticsImageFilter()
        cc = sitk.ConnectedComponent(array_cc)
        stats.Execute(cc, array_cc)

        (min_x, max_x, min_y, max_y, min_z, max_z) = stats.GetBoundingBox(1)

        if ((max_z-min_z) > 1):
            max_x = max_x+2
            max_y = max_y+2
            max_z = max_z+1

            # GT
            bounding_gt = gt_transpose[min_x:max_x, min_y:max_y, min_z:max_z]
            bounding_gt_array = sitk.GetArrayFromImage(bounding_gt)
            gt_original_array[min_z:max_z, min_y:max_y,
                              min_x:max_x] = bounding_gt_array

    return gt_original_array
