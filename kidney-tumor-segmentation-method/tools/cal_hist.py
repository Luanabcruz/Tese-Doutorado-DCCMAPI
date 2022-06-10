import os
import numpy as np
import matplotlib.pyplot as plt

# path = r'C:\Users\domin\Downloads\imagens_nova_dist_3007\imagens_nova_dist_3007\treino\Images\case_00000-0.png'
path = r'C:\Users\domin\Downloads\case_00097.npy'

# importing required libraries of opencv
import cv2
  
# importing library for plotting
from matplotlib import pyplot as plt
  
# reads an input image
# img = cv2.imread(path,0)
img = np.load(path)
# img = np.where(img == 0, np.nan, img)

# img = np.where(img < 66, np.nan, img)
# img = np.where(img > 86, np.nan, img)


plt.hist(img.ravel(),256,[60,90])
# plt.imshow(img[0],cmap='gray')
plt.show()