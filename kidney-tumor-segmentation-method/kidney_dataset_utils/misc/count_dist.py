'''
Script que realiza a contagem dos grupos dado um distribuição de casos
'''

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


def count_dataset_group():

    metadata = "./configuration_files/kits19_dist_base_v2.csv"  # Meta info

    cases = load_dataset_dist(dataset_id="natal_v7")
    train_valid_cases = []

    train_valid_cases.extend(cases['train'])
    train_valid_cases.extend(cases['valid'])

    df = pd.read_csv(metadata)

    groups = ['big-dark', 'big-light', 'medium-dark',
              'medium-light', 'tiny-dark', 'tiny-light']

    count = {}

    for group in groups:
        case_group_list = df.loc[df['groups']
                                 == group]['case'].tolist()

        intersection_set = set.intersection(
            set(case_group_list), set(train_valid_cases))

        intersection_list = list(intersection_set)
        count[group] = len(intersection_list)
        print('list:', len(intersection_list))

    print(count)


if __name__ == '__main__':
    count_dataset_group()
