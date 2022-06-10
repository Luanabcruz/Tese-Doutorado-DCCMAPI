import json
import os
import numpy as np
import random


def load_dataset_dist(path=None, dataset_id="lua_nova"):
    if path == None:
        path = './configuration_files/dataset_dist_fixed'

    with open(os.path.join(path, 'cases_division_{}.json'.format(dataset_id)), 'r') as f:
        dataset = json.load(f)

    return dataset


cases = load_dataset_dist(None,"natal_v7")

root_dir = r'D:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_tumor_masks\KiTS19_master_npz'

cases = cases['valid'] 

slices_list = []
all_tumors = 0
all_no_tumors = 0
all_slices  = 0
for case in cases:
    image_list = os.listdir(os.path.join(root_dir, case, 'Images'))

    slices_tumors = []
    slices_notumors = []

    for im_name in image_list:
        
        all_slices += 1

        mask = np.load(os.path.join(
            root_dir, case, 'Masks', 'masc_{}'.format(im_name)))
        mask = mask['arr_0']
        if mask.sum() > 0:
            slices_tumors.append(im_name)
        else:
            slices_notumors.append(im_name)

    num_slices_tumors = len(slices_tumors)
    
    

    # if num_slices_tumors < len(slices_notumors):
    #     slices_notumors = random.sample(slices_notumors, num_slices_tumors)

    print('{} , {}, {}'.format(case,
                                                len(slices_tumors), len(slices_notumors)))

    all_tumors += len(slices_tumors)
    all_no_tumors += len(slices_notumors)


def func(name):
    num = int(''.join(x for x in name if x.isdigit()))
    print(num)
    return num


print('\n\nKits19 dataset imbalance => {} positives {} negatives'.format(
    all_tumors, all_no_tumors))

print("\n Total de fatias: {}".format(all_slices))