from functools import reduce
import pandas as pd
import shutil
import os
import json
import random


def load_dataset_dist(path=None, dataset_id=2):
    if path == None:
        path = './configuration_files/dataset_dist_fixed'

    with open(os.path.join(path, 'cases_division_{}.json'.format(dataset_id)), 'r') as f:
        dataset = json.load(f)

    return dataset


def load_dpa_dist(filename):
    path = './configuration_files/dpa'

    with open(os.path.join(path, '{}'.format(filename)), 'r') as f:
        dataset = json.load(f)

    return dataset



def split_dataset():

    metadata = "./kidney_dataset_utils/kits19_dist_base_v3.csv"  # Meta info

    cases = ["case_{:05d}".format(i) for i in range(0, 210)]

    df = pd.read_csv(metadata)

    groups = ['big-dark', 'big-light', 'medium-dark',
              'medium-light', 'tiny-dark', 'tiny-light']

    train_list = []
    valid_list = []
    test_list = []

    # black_list = ['case_00069', 'case_00165']
    black_list = ['case_00069', 'case_00165']

    _all = []
    _all.extend(black_list)

    # for group in groups:

    #     case_group_list = df.loc[df['groups']
    #                              == group]['case'].tolist()
    #     total = len(case_group_list)
    #     case_group_list = [t for t in case_group_list if t not in _all]

    #     num_test = round(total*0.15)
    #     test = random.sample(case_group_list, num_test)
    #     test_list.extend(test)
    #     _all.extend(test)

    # if len(test_list) > 31:
    #     diff = len(test_list)
    #     test_list = test_list[:31-diff]
    #     _all = _all[:31-diff]

    test_list = [
        "case_00009",
        "case_00011",
        "case_00012",
        "case_00018",
        "case_00024",
        "case_00025",
        "case_00028",
        "case_00041",
        "case_00043",
        "case_00045",
        "case_00056",
        "case_00066",
        "case_00076",
        "case_00078",
        "case_00082",
        "case_00085",
        "case_00099",
        "case_00100",
        "case_00102",
        "case_00122",
        "case_00131",
        "case_00139",
        "case_00149",
        "case_00158",
        "case_00170",
        "case_00174",
        "case_00175",
        "case_00176",
        "case_00197",
        "case_00199",
        "case_00200"
    ]

    _all.extend(test_list)

    for group in groups:
        case_group_list = df.loc[df['groups']
                                 == group]['case'].tolist()
        total = len(case_group_list)
        case_group_list = [t for t in case_group_list if t not in _all]

        num_valid = round(total*0.15)
        valid = random.sample(case_group_list, num_valid)
        valid_list.extend(valid)

        _all.extend(valid)

    if len(valid_list) > 31:
        diff = len(valid_list)
        valid_list = valid_list[:31-diff]
        _all = _all[:31-diff]

    train_list = [t for t in cases if t not in _all]

    train_list.extend(black_list)

    train_list = list(set(sorted(train_list)))
    valid_list = list(set(sorted(valid_list)))
    test_list = list(set(sorted(test_list)))

    return train_list, valid_list, test_list


if __name__ == '__main__':
    train, valid, test = split_dataset()

    ds = {
        "train": sorted(train),
        "test": sorted(test),
        "valid": sorted(valid)
    }

    json = json.dumps(ds)

    f = open(
        "./configuration_files/dataset_dist_fixed/cases_division_natal_v8_asd.json", "w")
    f.write(json)
    f.close()

    print(len(train))
    print(len(valid))
    print(len(test))
