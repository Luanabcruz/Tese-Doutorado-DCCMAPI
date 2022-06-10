import os
import SimpleITK as sitk
import numpy as np
import json


def load_sitk(file_path, transpose=True):
    image = sitk.ReadImage(file_path)
    npy = sitk.GetArrayFromImage(image)

    if transpose:
        npy = npy.transpose(2, 0, 1)

    return npy, image


def create_nested_dir(path, destroy_dir=True):
    # Verifica a existência do diretório, para criá-lo caso não exista.
    if not os.path.exists(path):
        os.makedirs(path)


def gen_out_filename(dataset_type, case_name, index, output_dir, is_mask=False):

    file_template = ''

    if is_mask:
        path = os.path.join(output_dir, dataset_type, 'Masks')
        file_template = 'masc_{}-{}'.format(case_name, index)
    else:
        path = os.path.join(output_dir, dataset_type, 'Images')
        file_template = '{}-{}'.format(case_name, index)
    # cria diretório caso não exista
    create_nested_dir(path)

    return os.path.join(path, file_template)


def save_image_npz(image, mask, case_name, out_dir):

    empty_image = np.zeros([image.shape[1], image.shape[2]], dtype=np.uint8)
    channel = 3

    image = np.insert(image, 0, empty_image, axis=0)
    image = np.concatenate(
        (image, np.expand_dims(empty_image, axis=0)), axis=0)

    for i in range(0, image.shape[0]-channel+1):
        img_name = gen_out_filename(
            case_name, case_name, i, out_dir, is_mask=False)

        new_vol = None

        for j in range(0, channel):
            if new_vol is None:
                new_vol = np.expand_dims(
                    np.copy(image[i+j]), axis=0)
            else:
                new_vol = np.concatenate(
                    (new_vol, np.expand_dims(image[i+j], axis=0)), axis=0)

        np.savez_compressed(img_name, new_vol)

    if mask is not None:
        for i in range(0, mask.shape[0]):

            mask_name = gen_out_filename(
                case_name, case_name, i, out_dir, is_mask=True)

            np.savez_compressed(mask_name, mask[i])


def save_params(out_dir, params):

    with open(os.path.join(out_dir, "params.json"), 'w') as fp:
        json.dump(params, fp)
