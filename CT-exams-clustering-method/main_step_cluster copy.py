import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv ('out/features/feats_resnet18_concat_interpoleted.csv', sep=';')

cases = df['case_name'].tolist()

df = df.drop(columns='case_name')
df = df.astype(np.float32)
df = df.fillna(0)



# fitting multiple k-means algorithms and storing the values in an empty list
# SSE = []
# for cluster in range(2,15):
#     kmeans = KMeans( n_clusters = cluster, init='k-means++')
#     kmeans.fit(df)
#     SSE.append(kmeans.inertia_)

# # converting the results into a dataframe and plotting them
# frame = pd.DataFrame({'Cluster':range(2,15), 'SSE':SSE})
# plt.figure(figsize=(12,6))
# plt.plot(frame['Cluster'], frame['SSE'], marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()


from sklearn.metrics import silhouette_score


KMAX = 20
sil = []
# # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
# for k in range(2, kmax+1):
#   kmeans = KMeans(n_clusters = k).fit(df)
#   labels = kmeans.labels_
#   sil.append(silhouette_score(df, labels, metric = 'euclidean'))
# frame = pd.DataFrame({'Cluster':range(2, kmax+1), 'SIL':sil})
# plt.figure(figsize=(12,6))
# plt.plot(frame['Cluster'], frame['SIL'], marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('s-score')
# plt.show()

# https://jtemporal.com/kmeans-and-elbow-method/
# https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
# kmeans-and-elbow-method
def calculate_wcss(data):
        wcss = []
        for n in range(2, KMAX+1):
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(X=data)
            wcss.append(kmeans.inertia_)
    
        return wcss
wcss = calculate_wcss(df)

# frame = pd.DataFrame({'Cluster':range(2, KMAX+1), 'wcss':wcss})
# plt.figure(figsize=(12,6))
# plt.plot(frame['Cluster'], frame['wcss'], marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('wcss')
# plt.show()

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

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

kmeans = KMeans(n_clusters=6, init='k-means++').fit(df)
print(kmeans.labels_)

# from collections import Counter
# print(Counter(kmeans.labels_))
