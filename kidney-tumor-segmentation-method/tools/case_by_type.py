import os
import json
import SimpleITK as sitk
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import ttest_ind
from tqdm import tqdm

model_1 = r'./out/nii_dart_pos'
model_2 = r'./out/nii_sem_data_aug'

dices = []

kappas = []

y1 = []
y2 = []

for pred in tqdm(os.listdir(model_1)):

    sitk_model_1 = sitk.ReadImage(os.path.join(model_1, pred))
    sitk_model_1_array = sitk.GetArrayFromImage(sitk_model_1)

    sitk_model_2 = sitk.ReadImage(os.path.join(model_2, pred))
    sitk_model_2_array = sitk.GetArrayFromImage(sitk_model_2)

    # if len(y1) == 0:
    #     y1 = sitk_model_1_array.flatten().tolist()
    # else:    
    #     y1.extend(sitk_model_1_array.flatten().tolist())
    
    
    # if len(y2) == 0:
    #     y2 = sitk_model_2_array.flatten().tolist()
    # else:    
    #     y2.extend(sitk_model_2_array.flatten().tolist())
    

    # k = cohen_kappa_score(sitk_model_1_array.flatten(),sitk_model_2_array.flatten())
    k = ttest_ind(sitk_model_1_array.flatten(),sitk_model_2_array.flatten()).pvalue
    # print("{}: {}".format(pred, k))
    kappas.append(k)

kappas	= np.asarray(kappas)

print("p value médio", kappas.mean())

# pvalue = ttest_ind(y1,y2).pvalue

# print("P-value: ", pvalue)


    