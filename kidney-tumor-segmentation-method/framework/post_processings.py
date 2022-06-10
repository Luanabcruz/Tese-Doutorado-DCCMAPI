import numpy as np
import matplotlib.pyplot as plt
# import cv2
import SimpleITK as sitk
# import cc3d


def median_filter(vol):

    vol = vol.astype(np.uint8)

    sitk_image = sitk.GetImageFromArray(vol)
    sitk_median = sitk.BinaryMedianImageFilter()

    sitk_image = sitk_median.Execute(sitk_image)

    vol = sitk.GetArrayFromImage(sitk_image)

    return vol


def connected_components(vol):
    vol_ = vol.copy()
    vol_[vol_ > 0] = 1
    vol_cc = cc3d.connected_components(vol_)
    cc_sum = [(i, vol_cc[vol_cc == i].shape[0])
              for i in range(vol_cc.max() + 1)]
    cc_sum.sort(key=lambda x: x[1], reverse=True)
    cc_sum.pop(0)  # remove background
    reduce_cc = [cc_sum[i][0] for i in range(
        1, len(cc_sum)) if cc_sum[i][1] < cc_sum[0][1] * 0.1]
    for i in reduce_cc:
        vol[vol_cc == i] = 0

    return vol

# função de Lua para fazer um pós processamento analisando o caso em 3D


def min_slices_number_3D(gt_array):

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


# função de Lua para fazer um pós processamento fechamento 3D
def morphological_closing_3D(pred_image, kernel_value=1):

    pred_image = pred_image.astype(np.uint8)
    pred_image = sitk.GetImageFromArray(pred_image)

    morphological_c = sitk.BinaryMorphologicalClosingImageFilter()
    morphological_c.SetForegroundValue(1)
    morphological_c.SetKernelRadius(kernel_value)
    morphological_c.SetKernelType(2)
    morphological = morphological_c.Execute(pred_image)
    morphological = sitk.GetArrayFromImage(morphological)

    return morphological


def two_biggest_elem(pred_image):

    X_pred = pred_image
    X_pred = X_pred.astype(np.uint8)
    X_pred = sitk.GetImageFromArray(X_pred)

    stats = sitk.LabelIntensityStatisticsImageFilter()
    cc = sitk.ConnectedComponent(X_pred)

    stats.Execute(cc, X_pred)

    primeiro_label = None
    segundo_label = None
    size_primeiro_label = 0
    size_segundo_label = 0

    for l in stats.GetLabels():
        if (stats.GetNumberOfPixels(l) > size_primeiro_label):
            if(size_primeiro_label > size_segundo_label):
                size_segundo_label = size_primeiro_label
                segundo_label = primeiro_label
            size_primeiro_label = stats.GetNumberOfPixels(l)
            primeiro_label = l
        elif (stats.GetNumberOfPixels(l) > size_segundo_label):
            size_segundo_label = stats.GetNumberOfPixels(l)
            segundo_label = l

    # print(stats.GetNumberOfPixels(primeiro_label))
    # print(stats.GetNumberOfPixels(segundo_label))

    array_cc = sitk.GetArrayFromImage(cc)
    array_cc = ((array_cc == primeiro_label)*1) + \
        ((array_cc == segundo_label)*1)

    return array_cc
