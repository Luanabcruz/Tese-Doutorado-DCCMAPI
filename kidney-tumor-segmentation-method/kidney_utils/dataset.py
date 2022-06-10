

import os
import numpy as np
from PIL import Image


def load_casebyname_from_image_folder(folder_path, case_name, image_folder='Images', mask_folder='Masks', tensor_shape=True):
    case = {
        'image': [],
        'mask': [],
        'case_name': [],
        'mask_name': []
    }

    # primeiro, pego a lista de todas as fatias do caso informado (case code)

    files = os.listdir(os.path.join(folder_path, case_name, image_folder))
    case_filenames = files

    # funcao necessária pois quero que as fatias estejam ordenadas

    def func(name):
        num = int(''.join(x for x in name if x.isdigit()))
        return num

    case_filenames = sorted(case_filenames, key=func)

    for case_filename in case_filenames:

        # im = Image.open(os.path.join(
        #     folder_path, case_name, image_folder, case_filename))
        image_filename_fullpath = os.path.join(folder_path, case_name,
                                               image_folder, case_filename)

        im = np.load(image_filename_fullpath)

        __, file_extension = os.path.splitext(image_filename_fullpath)

        if file_extension == '.npz':
            im = im['arr_0']

        # if im.ndim == 2:
        #     im = np.stack((im,)*3, axis=-1)
        #     im = im.transpose(2, 1, 0)

        if im.ndim == 2:
            im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=0)
        if mask_folder is not None:
            mask_filename_fullpath = os.path.join(
                folder_path, case_name, mask_folder, 'masc_'+case_filename)

            mask = np.load(mask_filename_fullpath)

            if file_extension == '.npz':
                mask = mask['arr_0']

        case['image'].append(im)

        if mask_folder is not None:
            case['mask'].append(mask)
            case['mask_name'].append('masc_'+case_filename)

        case['case_name'].append(case_filename)

    case['image'] = np.asarray(case['image'])
    case['mask'] = np.asarray(case['mask'], dtype=np.uint8)

    return case


def load_case_from_image_folder(folder_path, case_code, image_folder='Images', mask_folder='Masks', tensor_shape=True):
    case = {
        'image': [],
        'mask': [],
        'case_name': [],
        'mask_name': []
    }

    # primeiro, pego a lista de todas as fatias do caso informado (case code)
    filename_template = "case_{:05d}-"
    files = os.listdir(os.path.join(folder_path, image_folder))
    case_filenames = []
    for filename in files:
        if filename_template.format(case_code) in filename:
            case_filenames.append(filename)

    # funcao necessária pois quero que as fatias estejam ordenadas

    def func(name):
        num = int(''.join(x for x in name if x.isdigit()))
        return num

    case_filenames = sorted(case_filenames, key=func)

    for case_name in case_filenames:

        im = Image.open(os.path.join(folder_path, image_folder, case_name))
        mask = np.asarray(Image.open(os.path.join(
            folder_path, mask_folder, 'masc_'+case_name)))

        if tensor_shape:
            im = im.convert('RGB')
            im = np.array(im).transpose(2, 0, 1).reshape(
                1, 3, im.width, im.height)

        case['image'].append(im)
        case['mask'].append(mask)
        case['case_name'].append(case_name)
        case['mask_name'].append('masc_'+case_name)

    case['image'] = np.asarray(case['image'])
    case['mask'] = np.asarray(case['mask'], dtype=np.uint8)

    return case


def load_case_mask_folder(folder_path, case_code):

    masks_array = []

    # primeiro, pego a lista de todas as fatias do caso informado (case code)
    filename_template = "case_{:05d}-"
    files = os.listdir(folder_path)
    case_filenames = []
    for filename in files:
        if filename_template.format(case_code) in filename:
            case_filenames.append(filename)

    # funcao necessária pois quero que as fatias estejam ordenadas

    def func(name):
        num = int(''.join(x for x in name if x.isdigit()))
        return num

    case_filenames = sorted(case_filenames, key=func)

    for case_name in case_filenames:

        mask = np.asarray(Image.open(os.path.join(
            folder_path, case_name)))

        mask = np.array(np.where(mask == 255, 1, 0), np.uint8)

        masks_array.append(mask)

    masks_array = np.asarray(masks_array)

    return masks_array


if __name__ == '__main__':
    load_case_from_image_folder('./kits19_lua/Test', 9)
