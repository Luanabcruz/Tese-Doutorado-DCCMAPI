import pandas as pd
import random
import json
import os

def delete_rand_items(items,n):
    to_delete = set(random.sample(range(len(items)),n))
    return [x for i,x in enumerate(items) if not i in to_delete]

def generate(index):

    str_template = "DPA_group_k{}"
    out_filename = str_template.format(index)
    
    filename = 'groups_resnet34_concat_interpoleted_k{}'.format(index)
    print(filename)
    df = pd.read_csv ('out/clusters/{}.csv'.format(filename), sep=';')

    TRAIN_RATIO = 148/179
    VALID_RATIO = 1 - TRAIN_RATIO

    groups_len = len(df['groups'].unique())

    # black_list = ['case_00069', 'case_00165']

    already_selected = []
    # already_selected.extend(black_list)

    valid_list = []

    for group_index in range(0, groups_len):
        case_by_group_list = df.loc[df['groups']== group_index]['cases'].tolist() 
        total_cases_by_group = len(case_by_group_list)

        case_by_group_list = [t for t in case_by_group_list if t not in already_selected]
        valid_len = round(VALID_RATIO * total_cases_by_group )
    
        cases_by_group_random = random.sample(case_by_group_list, valid_len)
        valid_list.extend(cases_by_group_random)
        already_selected.extend(cases_by_group_random)


    if len(valid_list) > 31:
        diff = len(valid_list) - 31
        valid_list = delete_rand_items(valid_list, diff)

    old_train_valid = df['cases'].to_list()

    valid_list = sorted(valid_list)
    train_list = sorted([x for x in old_train_valid if x not in valid_list])

    ds = {
        "train": sorted(train_list),
        "valid": sorted(valid_list)
    }

    json_data = json.dumps(ds)

    out_dir = './out/dpa/'
    try:
        os.makedirs(out_dir)
    except:
        pass  

    f = open(os.path.join(out_dir,"{}.json".format(out_filename)), "w")
    f.write(json_data)
    f.close()

if __name__ == '__main__':
    
    for i in range(1, 7):
        generate(i)
        


