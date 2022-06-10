import os
import json

def case2name(case):
    return ["case_{:05d}".format(i) for i in case]

train = sorted([0,1,2,3,4,5,6,7,10,12,13,14,15,17,19,21,22,23,24,25,27,30,31,32,33,35,36,37,38,39,40,42,44,46,47,48,49,50,51,52,53,54,56,57,58,59,61,62,63,64,67,68,71,73,75,77,80,81,83,84,88,89,90,91,92,93,94,95,96,98,99,102,104,105,106,107,109,110,111,112,113,114,116,118,119,121,123,124,125,126,127,128,132,133,134,135,137,138,140,141,143,144,145,146,147,148,150,151,152,153,154,155,156,159,162,163,164,166,167,168,169,171,172,173,175,177,178,179,181,182,185,186,188,189,192,193,195,196,198,201,202,203,204,205,206,207,209,69])
valid = sorted([79,55,187,8,70,157,130,29,184,183,16,142,180,108,34,20,161,86,87,208,136,120,101,72,60,190,194,74,117,65,115])
test = sorted([9,11,18,26,28,41,43,45,66,76,78,82,85,97,100,103,122,129,131,139,149,158,160,165,170,174,176,191,197,199,200,])

cases_division = {
    'train':case2name(train),
    'valid':case2name(valid),
    'test':case2name(test),
    'data_set_id':'lua_2'
}


with open(os.path.join('.', 'cases_division_lua_2'+".json"), 'w') as fp:
    json.dump(cases_division, fp)