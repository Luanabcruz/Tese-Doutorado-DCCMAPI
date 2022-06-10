import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

filename = 'feats_vgg19_concat_interpoleted'
df = pd.read_csv ('out/features/{}.csv'.format(filename), sep=';')

cases = df['case_name'].tolist()

df = df.drop(columns='case_name')
df = df.astype(np.float32)
df = df.fillna(0)

KMIN = 2
KMAX = 20

def calculate_wcss(data):
        wcss = []
        for n in range(KMIN, KMAX+1):
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(X=data)
            wcss.append(kmeans.inertia_)
    
        return wcss

wcss = calculate_wcss(df)

def optimal_number_of_clusters(wcss):
    x1, y1 = KMIN, wcss[0]
    x2, y2 = KMAX, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

clusters_indicados = optimal_number_of_clusters(wcss)
print("Numero indicado de clusters: ", clusters_indicados)

kmeans = KMeans(n_clusters=clusters_indicados, init='k-means++').fit(df)

from collections import Counter
print(Counter(kmeans.labels_))
print("Intertia: ", kmeans.inertia_)
new_dataframe = pd.DataFrame({'cases':cases, 'groups':kmeans.labels_})
out_dir = './out/clusters/'
try:
    os.makedirs(out_dir)
except:
    pass    
new_dataframe.to_csv('{}/{}.csv'.format(out_dir, filename), sep=';', decimal='.', index=False)


