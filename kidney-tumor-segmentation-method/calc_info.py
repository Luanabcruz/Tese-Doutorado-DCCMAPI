import os 
import numpy as np
import framework.ct_dataset_manager as dm

dataset = r'D:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_kidney_masks\Kits19_npz_25D_so_rim'

cases = dm.load_dataset_dist(dataset_id="natal_v7")['test']

len_all_slices = 0
len_all_pixels = 0
num_cases = len(cases)

for case in ["CASE_00009","case_00176"]:
    image_list = os.listdir(os.path.join(dataset, case,'Images'))
    len_all_slices += len(image_list)

print("Num cases: ", num_cases)   
print("Num de fatias: ", len_all_slices)
# print("Num de pixels em 256px:", len_all_slices*256*256)
# print("Num de pixels em 512px:", len_all_slices*512*512)