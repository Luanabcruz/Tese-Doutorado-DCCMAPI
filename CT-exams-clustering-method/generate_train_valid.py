import random
import os 
import json

out_dir = './config/'
out_filename = "cases_train_valid_k6"

all_cases = ["case_{:05d}".format(i) for i in range(0, 210)]
black_list =  ["case_{:05d}".format(i) for i in [0, 1, 18, 43, 85, 100, 122, 165]] 

cases = [case for case in all_cases if case not in black_list]

test_len = 31

test_cases = random.sample(cases, test_len)

ds = {
    "train_valid": sorted(cases),
    "test": sorted(test_cases)
}

json_data = json.dumps(ds)

f = open(os.path.join(out_dir,"{}.json".format(out_filename)), "w")
f.write(json_data)
f.close()