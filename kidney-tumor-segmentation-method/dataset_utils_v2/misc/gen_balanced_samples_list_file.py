import json
import os
import numpy as np
import random

root_dir = r'D:\DOUTORADO_LUANA\bases25D\KiTS19_master_npy'

cases = os.listdir(root_dir)
# case = cases[165]

slices_list = []

for case in cases:
    image_list = os.listdir(os.path.join(root_dir, case, 'Images'))

    # print(image_list)
    slices_tumors = []
    slices_notumors = []

    for im_name in image_list:
        mask = np.load(os.path.join(
            root_dir, case, 'Masks', 'masc_{}'.format(im_name)))

        if mask.sum() > 0:
            slices_tumors.append(im_name)
        else:
            slices_notumors.append(im_name)

    num_slices_tumors = len(slices_tumors)

    if num_slices_tumors < len(slices_notumors):
        slices_notumors = random.sample(slices_notumors, num_slices_tumors)

    slices_list.extend(slices_notumors)
    slices_list.extend(slices_tumors)

    print('{} => {} tumors {} no tumors'.format(case,
                                                len(slices_tumors), len(slices_notumors)))


def func(name):
    num = int(''.join(x for x in name if x.isdigit()))
    print(num)
    return num


slices_list = sorted(slices_list, key=func)

with open('balanced_slices.json', 'w') as f:
    json.dump(slices_list, f)
