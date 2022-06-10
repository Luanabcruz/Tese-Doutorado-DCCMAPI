import pandas as pd
import math
import functools
import operator

nn = 'resnet34'
filename = 'feats_{}_concat_interpoleted'.format(nn)
clusters_indicados = 6

df = pd.read_csv ('out/clusters/{}.csv'.format(filename), sep=';')

n_groups = len(df.groupby('groups')["cases"].count())

total_cases_train_val = 179
val_cases = 31

valid_dist_quant_by_group = {}

for idx_group in range (0, n_groups):
    quant = df.groupby('groups')["cases"].count()[idx_group]
    valid_dist_quant_by_group[idx_group] =  math.ceil(val_cases * quant/total_cases_train_val)
    print(quant, quant/total_cases_train_val, math.ceil(val_cases * quant/total_cases_train_val))