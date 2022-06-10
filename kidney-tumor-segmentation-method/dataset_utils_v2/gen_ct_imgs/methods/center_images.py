import SimpleITK as sitk
import numpy as np
import cv2


def centralizar_imagens(bounding_vl, bounding_gt, width=256, height=256):

    image_total = np.zeros((bounding_vl.shape[0], width, height))
    gt_total = np.zeros((bounding_vl.shape[0], width, height))

    imagem_menor = bounding_vl
    gt_menor = bounding_gt

    min_x = (image_total.shape[0]-imagem_menor.shape[0])//2
    min_y = (image_total.shape[1]-imagem_menor.shape[1])//2
    min_z = (image_total.shape[2]-imagem_menor.shape[2])//2

    image_total[min_x:min_x+imagem_menor.shape[0], min_y:min_y +
                imagem_menor.shape[1], min_z:min_z+imagem_menor.shape[2]] = imagem_menor
    gt_total[min_x:min_x+imagem_menor.shape[0], min_y:min_y +
             imagem_menor.shape[1], min_z:min_z+imagem_menor.shape[2]] = gt_menor

    return image_total, gt_total


def descentralizar_imagens(image, ground, target_shape):

    min_x = (image.shape[0]-target_shape.shape[0])//2
    min_y = (image.shape[1]-target_shape.shape[1])//2
    min_z = (image.shape[2]-target_shape.shape[2])//2

    image_out = image[min_x:min_x+target_shape.shape[0], min_y:min_y +
                      target_shape.shape[1], min_z:min_z+target_shape.shape[2]]
    ground_out = ground[min_x:min_x+target_shape.shape[0], min_y:min_y +
                        target_shape.shape[1], min_z:min_z+target_shape.shape[2]]

    return image_out, ground_out


def descentralizar_gt(ground, target_shape):

    tx = target_shape[0]
    ty = target_shape[1]
    tz = target_shape[2]

    min_x = (ground.shape[0]-tx)//2
    min_y = (ground.shape[1]-ty)//2
    min_z = (ground.shape[2]-tz)//2

    ground_out = ground[min_x:min_x+tx, min_y:min_y+ty, min_z:min_z+tz]

    return ground_out


def centralizar_imagens_x_y(bounding_vl, bounding_gt, width=256, height=256):

    image_total = np.zeros((bounding_vl.shape[0], width, height))
    gt_total = np.zeros((bounding_vl.shape[0], width, height))

    imagem_menor = bounding_vl
    gt_menor = bounding_gt

    min_x = (image_total.shape[0]-imagem_menor.shape[0])//2
    min_y = (image_total.shape[1]-imagem_menor.shape[1])//2
    min_z = (image_total.shape[2]-imagem_menor.shape[2])//2

    image_total[min_x:min_x+imagem_menor.shape[0], min_y:min_y +
                imagem_menor.shape[1], min_z:min_z+imagem_menor.shape[2]] = imagem_menor
    gt_total[min_x:min_x+imagem_menor.shape[0], min_y:min_y +
             imagem_menor.shape[1], min_z:min_z+imagem_menor.shape[2]] = gt_menor

    return image_total, gt_total


def centralizar_imagens_x(bounding_vl, bounding_gt, dimension=256):

    image_total = np.zeros(
        (bounding_vl.shape[0], dimension, bounding_vl.shape[2]))
    gt_total = np.zeros(
        (bounding_vl.shape[0], dimension, bounding_vl.shape[2]))

    imagem_menor = bounding_vl
    gt_menor = bounding_gt

    min_x = (image_total.shape[0]-imagem_menor.shape[0])//2
    min_y = (image_total.shape[1]-imagem_menor.shape[1])//2
    min_z = (image_total.shape[2]-imagem_menor.shape[2])//2

    image_total[min_x:min_x+imagem_menor.shape[0], min_y:min_y +
                imagem_menor.shape[1], min_z:min_z+imagem_menor.shape[2]] = imagem_menor
    gt_total[min_x:min_x+imagem_menor.shape[0], min_y:min_y +
             imagem_menor.shape[1], min_z:min_z+imagem_menor.shape[2]] = gt_menor

    return image_total, gt_total


def centralizar_imagens_y(bounding_vl, bounding_gt, dimension=256):

    image_total = np.zeros((bounding_vl.shape[0], bounding_vl.shape[1], 256))
    gt_total = np.zeros((bounding_vl.shape[0], bounding_vl.shape[1], 256))

    imagem_menor = bounding_vl
    gt_menor = bounding_gt

    min_x = (image_total.shape[0]-imagem_menor.shape[0])//2
    min_y = (image_total.shape[1]-imagem_menor.shape[1])//2
    min_z = (image_total.shape[2]-imagem_menor.shape[2])//2

    image_total[min_x:min_x+imagem_menor.shape[0], min_y:min_y +
                imagem_menor.shape[1], min_z:min_z+imagem_menor.shape[2]] = imagem_menor
    gt_total[min_x:min_x+imagem_menor.shape[0], min_y:min_y +
             imagem_menor.shape[1], min_z:min_z+imagem_menor.shape[2]] = gt_menor

    return image_total, gt_total
