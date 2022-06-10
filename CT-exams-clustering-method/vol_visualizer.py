import numpy as np
import os
import matplotlib.pyplot as plt
if __name__ == '__main__':

    path = './cache/tumor512_vol'
    case = "case_{:05d}.npz"
    case_number = 0
    case_data = np.load(os.path.join(path, case.format(case_number)))['arr_0']

    for i in range(0, case_data.shape[2]):
        print("Max: {} | Min: {} | Mean: {} ".format(case_data[:,:,i].max(), case_data[:,:,i].min(), case_data[:,:,i].mean()))        
        plt.imshow(case_data[:,:,i], cmap='gray')
        plt.show()




