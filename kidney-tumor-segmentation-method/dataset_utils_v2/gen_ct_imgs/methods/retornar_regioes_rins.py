import SimpleITK as sitk
import numpy as np
import cv2
import glob


def function_delimited_kidneys_challenge(gt_array, imagem_array):
    gt_array_rins = np.zeros(gt_array.shape)
    gt_array_rins[np.where(gt_array != 0)] = 1
    gt_array_rins = gt_array_rins.astype(np.uint8)

    gt_rins = sitk.GetImageFromArray(gt_array_rins)

    gt_transpose = sitk.GetImageFromArray(gt_array)  # classes originais
    imagem_transpose = sit.GetImageFromArray(imagem_array)

    stats = sitk.LabelStatisticsImageFilter()
    cc_intensity = sitk.ConnectedComponent(gt_rins)
    stats.Execute(cc_intensity, gt_rins)

    (min_x, max_x, min_y, max_y, min_z, max_z) = stats.GetBoundingBox(1)

    # normal
    max_x = max_x+2
    max_y = max_y+2
    max_z = max_z+1

    # porcentagem = 0.05
    # porc_min_max_x = int((max_x - min_x)*porcentagem)
    # porc_min_max_y = int((max_y - min_y)*porcentagem)
    # porc_min_max_z = int((max_z - min_z)*porcentagem)

    # min_x = min_x-porc_min_max_x
    # max_x = max_x+porc_min_max_x
    # min_y = min_y-porc_min_max_y
    # max_y = max_y+porc_min_max_y
    # min_z = min_z-porc_min_max_z
    # max_z = max_z+porc_min_max_z

    # GT
    bounding_gt = gt_transpose[min_x:max_x, min_y:max_y, min_z:max_z]
    bounding_gt_array = sitk.GetArrayFromImage(bounding_gt)

    # IMG
    bounding_image = imagem_tranpose[min_x:max_x, min_y:max_y, min_z:max_z]
    bounding_image_array = sitk.GetArrayFromImage(bounding_image)

    positions = [min_x, max_x, min_y, max_y, min_z, max_z]

    return positions, bounding_gt_array, bounding_image_array


def function_delimited(imagem_array, gt_array):
    gt_array_rins = np.zeros(gt_array.shape)
    gt_array_rins[np.where(gt_array != 0)] = 1
    gt_array_rins = gt_array_rins.astype(np.uint8)

    gt_rins = sitk.GetImageFromArray(gt_array_rins)

    volume_transpose = sitk.GetImageFromArray(imagem_array)
    gt_transpose = sitk.GetImageFromArray(gt_array)  # classes originais

    stats = sitk.LabelStatisticsImageFilter()
    cc_intensity = sitk.ConnectedComponent(gt_rins)
    stats.Execute(cc_intensity, gt_rins)

    (min_x, max_x, min_y, max_y, min_z, max_z) = stats.GetBoundingBox(1)

    # normal
    max_x = max_x+2
    max_y = max_y+2
    max_z = max_z+1

    # porcentagem = 0.05
    # porc_min_max_x = int((max_x - min_x)*porcentagem)
    # porc_min_max_y = int((max_y - min_y)*porcentagem)
    # porc_min_max_z = int((max_z - min_z)*porcentagem)

    # min_x = min_x-porc_min_max_x
    # max_x = max_x+porc_min_max_x
    # min_y = min_y-porc_min_max_y
    # max_y = max_y+porc_min_max_y
    # min_z = min_z-porc_min_max_z
    # max_z = max_z+porc_min_max_z

    positions = [min_x, max_x, min_y, max_y, min_z, max_z]

    return positions, volume_transpose, gt_transpose


def delimite_predict_origin(imagem_array, gt_array, gt_predict):

    positions_original, imagem_original, gt_original = function_delimited(
        imagem_array, gt_array)
    positions_prediction, imagem_prediction, gt_prediction = function_delimited(
        imagem_array, gt_predict)

    # min_x
    if(positions_prediction[0] > positions_original[0]):
        positions_prediction[0] = positions_original[0]
    else:
        positions_original[0] = positions_prediction[0]
    # max_x
    if(positions_prediction[1] < positions_original[1]):
        positions_prediction[1] = positions_original[1]
    else:
        positions_original[1] = positions_prediction[1]
    # min_y
    if(positions_prediction[2] > positions_original[2]):
        positions_prediction[2] = positions_original[2]
    else:
        positions_original[2] = positions_prediction[2]
    # max_y
    if(positions_prediction[3] < positions_original[3]):
        positions_prediction[3] = positions_original[3]
    else:
        positions_original[3] = positions_prediction[3]
    # min_z
    if(positions_prediction[4] > positions_original[4]):
        positions_prediction[4] = positions_original[4]
    else:
        positions_original[4] = positions_prediction[4]
    # max_z
    if(positions_prediction[5] < positions_original[5]):
        positions_prediction[5] = positions_original[5]
    else:
        positions_original[5] = positions_prediction[5]

    # GT, IMG ORIGINAL VOLUME
    gt_original_bb = gt_original[positions_original[0]:positions_original[1],
                                 positions_original[2]:positions_original[3], positions_original[4]:positions_original[5]]
    imagem_original_bb = imagem_original[positions_original[0]:positions_original[1],
                                         positions_original[2]:positions_original[3], positions_original[4]:positions_original[5]]

    # GT, IMG PREDICTION VOLUME
    gt_prediction_bb = gt_prediction[positions_prediction[0]:positions_prediction[1],
                                     positions_prediction[2]:positions_prediction[3], positions_prediction[4]:positions_prediction[5]]
    imagem_prediction_bb = imagem_prediction[positions_prediction[0]:positions_prediction[1],
                                             positions_prediction[2]:positions_prediction[3], positions_prediction[4]:positions_prediction[5]]

    # GT, IMG ORIGINAL ARRAY
    gt_original_bb_array = sitk.GetArrayFromImage(gt_original_bb)
    imagem_original_bb_array = sitk.GetArrayFromImage(imagem_original_bb)

    # GT, IMG PREDICTION ARRAY
    gt_prediction_bb_array = sitk.GetArrayFromImage(gt_prediction_bb)
    imagem_prediction_bb_array = sitk.GetArrayFromImage(imagem_prediction_bb)

    positions = [positions_original[0], positions_original[1], positions_original[2],
                 positions_original[3], positions_original[4], positions_original[5]]

    return positions, imagem_original_bb_array, gt_original_bb_array, imagem_prediction_bb_array, gt_prediction_bb_array


def back_image_original(gt_array, img_gt_original, positions):

    gt_total = np.zeros(img_gt_original)

    # GT
    #gt_aux = gt_transpose[min_x:max_x,min_y:max_y,min_z:max_z]
    #gt_aux = sitk.GetArrayFromImage(gt_aux)
    gt_total[positions[4]:positions[5], positions[2]:positions[3], positions[0]:positions[1]] = gt_array

    return gt_total
