import pandas as pd
import os
import json

def save_to_csv(columns, data, filename='feats',sep=';', decimal='.'):
    df = pd.DataFrame(columns=columns)
    df = df.append(data, ignore_index=True)
    try:
        os.makedirs('./out/')
    except:
        pass    
    df.to_csv('./out/{}.csv'.format(filename), sep=sep, decimal=decimal, index=False)
    
def cases_name_train_valid(k = None):
    dataset = {}
    prefix = ''
    
    if k is not None:
        prefix = '_k' + str(k)

    with open(os.path.join('config', 'cases_train_valid{}.json'.format(prefix)), 'r') as f:
        dataset = json.load(f)
    dataset['train_valid'] = sorted(dataset['train_valid'])
    return dataset['train_valid']

def cases_name_all():
    cases = ["case_{:05d}".format(i) for i in range(0, 210)]
    return cases    

