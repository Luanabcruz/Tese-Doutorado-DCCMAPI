import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from sklearn.metrics import silhouette_score

nn = 'resnet34'
filename = 'feats_{}_concat_interpoleted_k6'.format(nn)
clusters_indicados = 6

df = pd.read_csv ('out/features/{}.csv'.format(filename), sep=';')

cases = df['case_name'].tolist()

df = df.drop(columns='case_name')
df = df.astype(np.float32)
df = df.fillna(0)

kmeans = KMeans(n_clusters=clusters_indicados, init='k-means++')
preds = kmeans.fit_predict(df)
score = silhouette_score(df, preds)

from collections import Counter
print(Counter(kmeans.labels_))
print ("silhouette score is {}".format(score))
print("Intertia: ", kmeans.inertia_)
new_dataframe = pd.DataFrame({'cases':cases, 'groups':kmeans.labels_})
out_dir = './out/clusters/'
try:
    os.makedirs(out_dir)
except:
    pass    
new_dataframe.to_csv('{}/{}.csv'.format(out_dir, filename.replace("feats", "groups")), sep=';', decimal='.', index=False)


