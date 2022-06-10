import os
import numpy as np
import json

IMAGE_FOLDER = 'Images'
MASK_FOLDER = 'Masks'

# funcao necessária pois quero que as fatias estejam ordenadas


def order_slices_by_name(name):
    num = int(''.join(x for x in name if x.isdigit()))
    return num


def load_case_from_folder(folder_path, case_name):

    case = {
        'image': [],
        'mask': [],
        'case_name': [],
        'mask_name': []
    }

    # primeiro, pego a lista de todas as fatias do caso informado (case code)

    files = os.listdir(os.path.join(folder_path, case_name, IMAGE_FOLDER))
    case_filenames = files

    case_filenames = sorted(case_filenames, key=order_slices_by_name)

    for case_filename in case_filenames:
        image_filename_fullpath = os.path.join(folder_path, case_name,
                                               IMAGE_FOLDER, case_filename)

        im = np.load(image_filename_fullpath)

        __, file_extension = os.path.splitext(image_filename_fullpath)

        if file_extension == '.npz':
            im = im['arr_0']

        if im.ndim == 2:
            im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=0)
        if MASK_FOLDER is not None:
            mask_filename_fullpath = os.path.join(
                folder_path, case_name, MASK_FOLDER, 'masc_'+case_filename)

            mask = np.load(mask_filename_fullpath)

            if file_extension == '.npz':
                mask = mask['arr_0']

        case['image'].append(im)

        if MASK_FOLDER is not None:
            case['mask'].append(mask)
            case['mask_name'].append('masc_'+case_filename)

        case['case_name'].append(case_filename)

    case['image'] = np.asarray(case['image'])
    case['mask'] = np.asarray(case['mask'], dtype=np.uint8)

    return case


def load_dataset_dist(path=None, dataset_id=2):
    if path == None:
        path = './configuration_files/dataset_dist_fixed'

    with open(os.path.join(path, 'cases_division_{}.json'.format(dataset_id)), 'r') as f:
        dataset = json.load(f)

    return dataset
