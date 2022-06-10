import SimpleITK as sitk
import numpy as np
import cv2
import glob


def labels_kidney(gt_array):

    gt_array_rins = np.array(np.where(gt_array > 0, 1, 0), np.uint8)
    gt_array_rins = gt_array_rins.astype(np.uint8)

    gt_rins = sitk.GetImageFromArray(gt_array_rins)

    gt_transpose = sitk.GetImageFromArray(gt_array)  # classes originais

    stats = sitk.LabelIntensityStatisticsImageFilter()
    cc_intensity = sitk.ConnectedComponent(gt_rins)
    stats.Execute(cc_intensity, gt_rins)

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

    label = primeiro_label, segundo_label
    gt_original_array = np.zeros(gt_array.shape)

    min_y_esquerdo = 1000000
    min_y_direito = -1000000

    for ps_label in label:

        if(ps_label == None):
            continue

        array_cc = sitk.GetArrayFromImage(cc_intensity)
        array_cc = ((array_cc == ps_label)*1)

        array_cc = sitk.GetImageFromArray(array_cc)

        stats = sitk.LabelStatisticsImageFilter()
        cc = sitk.ConnectedComponent(array_cc)
        stats.Execute(cc, array_cc)

        (min_x, max_x, min_y, max_y, min_z, max_z) = stats.GetBoundingBox(1)

        max_x = max_x+2
        max_y = max_y+2
        max_z = max_z+1

        if(min_y < min_y_esquerdo):
            label_esquerdo = ps_label
            min_y_esquerdo = min_y
        if(min_y > min_y_direito):
            label_direito = ps_label
            min_y_direito = min_y

    return label_esquerdo, label_direito


def kidney_features(gt_array, label):

    gt_array_kidney = np.array(np.where(gt_array > 0, 1, 0), np.uint8)
    gt_rins = sitk.GetImageFromArray(gt_array_kidney)

    stats = sitk.LabelIntensityStatisticsImageFilter()
    cc_intensity = sitk.ConnectedComponent(gt_rins)
    stats.Execute(cc_intensity, gt_rins)

    array_cc = sitk.GetArrayFromImage(cc_intensity)
    array_cc = ((array_cc == label)*1)

    array_cc = sitk.GetImageFromArray(array_cc)

    stats_specific = sitk.LabelStatisticsImageFilter()
    cc = sitk.ConnectedComponent(array_cc)
    stats_specific.Execute(cc, array_cc)

    kidney_area = stats.GetNumberOfPixels(label)

    (min_x, max_x, min_y, max_y, min_z, max_z) = stats_specific.GetBoundingBox(1)
    # print(min_x,max_x,min_y,max_y,min_z,max_z)

    max_x = max_x+2
    max_y = max_y+2
    max_z = max_z+1

    kidney_slices = max_z - min_z

    gt_array_volume = export_volume(
        gt_array, min_x, max_x, min_y, max_y, min_z, max_z)

    return gt_array_volume, kidney_area, kidney_slices


def quantity_labels_lesion(gt):

    gt_array = np.array(np.where(gt < 2, 0, 2), np.uint8)
    gt_array = sitk.GetImageFromArray(gt_array)

    stats = sitk.LabelIntensityStatisticsImageFilter()
    cc_intensity = sitk.ConnectedComponent(gt_array)
    stats.Execute(cc_intensity, gt_array)

    labels = stats.GetLabels()

    return labels


def lesion_features(imagem_array, gt_array, label):

    gt_array_lesao = np.array(np.where(gt_array < 2, 0, 2), np.uint8)
    gt_rins = sitk.GetImageFromArray(gt_array_lesao)

    stats = sitk.LabelIntensityStatisticsImageFilter()
    cc_intensity = sitk.ConnectedComponent(gt_rins)
    stats.Execute(cc_intensity, gt_rins)

    array_cc = sitk.GetArrayFromImage(cc_intensity)
    array_cc = ((array_cc == label)*1)

    array_cc = sitk.GetImageFromArray(array_cc)

    stats_specific = sitk.LabelStatisticsImageFilter()
    cc = sitk.ConnectedComponent(array_cc)
    stats_specific.Execute(cc, array_cc)

    color_result = dark_or_light(
        imagem_array, gt_array_lesao, min_x, max_x, min_y, max_y, min_z, max_z)

    return lesion_area, lesion_slices, color_result

    (min_x, max_x, min_y, max_y, min_z, max_z) = stats_specific.GetBoundingBox(1)
    # print(min_x,max_x,min_y,max_y,min_z,max_z)

    max_x = max_x+2
    max_y = max_y+2
    max_z = max_z+1

    lesion_slices = max_z - min_z

    max_x = max_x+2
    max_y = max_y+2
    max_z = max_z+1

    # GT
    bounding_gt = gt_transpose[min_x:max_x, min_y:max_y, min_z:max_z]
    bounding_gt = sitk.GetArrayFromImage(bounding_gt)  # retorna só a BB
    # gt_original_array[min_z:max_z,min_y:max_y,min_x:max_x] = bounding_gt # retorna a BB na imagem original

    # VOLUME
    bounding_img = volume_transpose[min_x:max_x, min_y:max_y, min_z:max_z]
    bounding_img = sitk.GetArrayFromImage(bounding_img)
    # gt_original_array[min_z:max_z,min_y:max_y,min_x:max_x] = bounding_gt # retorna a BB na imagem original

    bounding_gt = np.array(np.where(bounding_gt > 0, 1, 0), np.uint8)

    p = np.nonzero(bounding_gt)
    dark = 0
    light = 0
    result = -1

    ids = []
    for x in range(len(p[0])):
        # ids.append(bounding_img[p[0][x],p[1][x],p[2][x]])
        if((bounding_img[p[0][x], p[1][x], p[2][x]]) < 405):
            light = light + 1
        else:
            dark = dark + 1

    if(dark > light):
        result = 0
    else:
        result = 1

    return result


def export_volume(gt_array, min_x, max_x, min_y, max_y, min_z, max_z):

    gt_transpose = sitk.GetImageFromArray(gt_array)
    gt_original_array = np.zeros(gt_array.shape)

    # GT
    bounding_gt = gt_transpose[min_x:max_x, min_y:max_y, min_z:max_z]
    bounding_gt = sitk.GetArrayFromImage(bounding_gt)  # retorna só a BB
    # retorna a BB na imagem original
    gt_original_array[min_z:max_z, min_y:max_y, min_x:max_x] = bounding_gt

    # gt_original_array = sitk.GetImageFromArray(gt_original_array)
    # sitk.WriteImage(gt_original_array, "gt_array.nii")

    return gt_original_array
