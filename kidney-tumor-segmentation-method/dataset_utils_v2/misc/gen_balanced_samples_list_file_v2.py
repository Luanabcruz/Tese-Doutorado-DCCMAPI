import json
import os
import numpy as np
import random

root_dir = r'D:\DOUTORADO_LUANA\bases25D\KiTS19_master_npy'

cases = os.listdir(root_dir)
# case = cases[165]

slices_list = []
# todas as fatias que não possuem lesão
slices_notumors_all = []
slices_tumors_all = []
balanced_notumors = []
balanced_tumors = []


def diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))


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
            slices_tumors_all.append(im_name)
        else:
            slices_notumors.append(im_name)
            slices_notumors_all.append(im_name)

    num_slices_tumors = len(slices_tumors)

    if num_slices_tumors < len(slices_notumors):
        slices_notumors = random.sample(slices_notumors, num_slices_tumors)

    balanced_notumors.extend(slices_notumors)
    balanced_tumors.extend(slices_tumors)

    slices_list.extend(slices_notumors)
    slices_list.extend(slices_tumors)

    print('{} => {} tumors {} no tumors'.format(case,
                                                len(slices_tumors), len(slices_notumors)))


print('{} => {} tumors {} no tumors'.format(
    'all cases', len(slices_tumors_all), len(slices_notumors_all)))


print('{} => {} tumors {} no tumors'.format(
    'balanced cases', len(balanced_tumors), len(balanced_notumors)))

if len(balanced_tumors) > len(balanced_notumors):
    diff_len = len(balanced_tumors) - len(balanced_notumors)

    diff_notumors_list = diff(slices_notumors_all, balanced_notumors)
    print('diff no turmors', len(diff_notumors_list))
    aux_notumors = random.sample(diff_notumors_list, diff_len)
    print('extra ', len(aux_notumors))
    slices_list.extend(aux_notumors)


def func(name):
    num = int(''.join(x for x in name if x.isdigit()))
    return num


slices_list = sorted(slices_list, key=func)
print('Total de fatias', len(slices_list))
with open('balanced_slicesv2.json', 'w') as f:
    json.dump(slices_list, f)
