import cv2
import numpy as np
from skimage.filters import unsharp_mask


def get_labels(preprocessings):
    labels = ['nenhum', 'bilateral', 'clahe',
              'unsharp_mask', 'tophat', 'eqHist']

    return [labels[pre] for pre in preprocessings]


def params_labels():
    return ['sigma_color', 'sigma_space', 'clip_limit', 'tile_grid', 'radius', 'amount']


def chromossome_labels(chromossome):
    return get_labels(chromossome[:3])+params_labels()


def chromossome_values(chromossome):
    return get_labels(chromossome[:3])+chromossome[3:]


def normalize(image):

    return (image - np.min(image))/255


def apply3(id, image, sigma_color, sigma_space, clip_limit, tile_grid, radius, amount):
    tile_grid = int(tile_grid)
    radius = int(radius)
    amount = int(amount)
    for i in range(0, image.shape[2]):
        image[:, :, i] = apply(
            id, image[:, :, i], sigma_color, sigma_space, clip_limit, tile_grid, radius, amount)

    return image


def apply(id, image, sigma_color=0.5, sigma_space=0.5, clip_limit=0.5, tile_grid=30, radius=1, amount=1):
    kernel = (3, 3)
    if id == 0:
        pass  # não aplica nada
    elif id == 1:
        image = np.asarray(image, np.float32)
        image = cv2.bilateralFilter(image, -1, sigma_color, sigma_space)
    elif id == 2:
        image *= 255
        image = np.asarray(image, np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(tile_grid, tile_grid))
        image = clahe.apply(image)
        image = normalize(image)
    elif id == 5:
        image *= 255
        image = np.asarray(image, np.uint8)
        image = cv2.equalizeHist(image)
        image = normalize(image)
    elif id == 4:

        filterSize = (3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           filterSize)
        image -= cv2.morphologyEx(image,
                                  cv2.MORPH_TOPHAT,
                                  kernel)
    elif id == 3:
        image = unsharp_mask(image, radius=radius, amount=amount)
    return image


def apply_preprocessing_list(image, params):
    list_pre = params[:3]
    pre_params = params[3:]
    for i in range(0, len(list_pre)):
        image = apply3(list_pre[i], image, sigma_color=pre_params[0],
                       sigma_space=pre_params[1], clip_limit=pre_params[2], tile_grid=pre_params[3], radius=pre_params[4], amount=pre_params[5])

    return image


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = r'E:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_tumor_masks\KiTS19_master_npz\case_00004\Images\case_00004-25.npz'

    im = np.load(path)
    im = im['arr_0']
    im = im.transpose(1, 2, 0)
    # im = apply(8, im)

    im = apply_preprocessing_list(
        im,  [4, 3, 4, 0.6830141089425715, 0.7784288771611926, 0.4676658223643955, 16, 6, 1])

    # print(im.max())
    # print(im.shape)
    plt.imshow(im[:, :, 0], cmap='gray')
    plt.show()
