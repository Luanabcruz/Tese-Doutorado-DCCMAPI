import matplotlib.pyplot as plt
import os
import numpy as np

channels = 3
case = "case_00009"
# outdir = r"D:\DOUTORADO_LUANA\test_kits19\npy"
# outdir = r"KiTS19_master_npy"
# outdir = r"D:\DOUTORADO_LUANA\test_kits19\npy"
# outdir = r"D:\DOUTORADO_LUANA\bases25D\KiTS19_master_npz"
# outdir = r'C:\Users\usuario\Documents\UFMA\DoutoradoLuana\segmentacao2d\datasets\KiTS19_master_npz'
outdir = r'D:\DoutoradoLuana\datasets\gerados\sem_balanceamento\pred_resunet25D_sem_balanc_256'

# dtype = "imaging"
dtype = "Images"
# dtype = "Masks"
# print(os.listdir(outdir))
for fname in os.listdir(os.path.join(outdir, case, dtype)):
    vol = np.load(os.path.join(outdir, case, dtype, fname))
    vol = vol['arr_0']
    # exit()
    # print(np.unique(vol))
    # exit()
    gt = np.load(os.path.join(outdir, case, "Masks",  "masc_"+fname))
    gt = gt['arr_0']

    # pula os vazios
    # if np.sum(gt) == 0:
    #     continue
    # print(vol.shape)
    # exit()
    if dtype == "Images" or dtype == "Images_Aug":
        fig, ax = plt.subplots(2, channels)
        fig.canvas.set_window_title(fname)
        # plt.imshow(gt, cmap='gray')

        for i in range(0, channels):
            ax[0][i].imshow(vol[i], cmap='gray')
            ax[0][i].axis('off')

        for i in range(0, channels):
            ax[1][i].axis('off')
            if i == int(channels//2):
                ax[1][i].imshow(gt, cmap='gray')

        # ax[0].imshow(vol, cmap='gray')
        # ax[0].axis('off')

        # ax[1].axis('off')

        # ax[1].imshow(gt, cmap='gray')

        # ax[1][2].imshow(gt, cmap='gray')
        # ax[1][2].axis('off')
        # ax[1][2].axis('off')
        # ax[1].imshow(vol[1], cmap='gray')
        # ax[1].axis('off')
        # ax[2].imshow(vol[2], cmap='gray')
        # ax[2].axis('off')

        plt.axis('off')
        plt.show()
        # break
        # plt.savefig("out_image//"+"masc_"+fname.replace('.npy', '.png'),
        #             transparent=True, bbox_inches='tight', pad_inches=0)

    else:
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title("case code: ")
        ax.imshow(vol, cmap='gray')
        ax.axis('off')

        plt.axis('off')
        plt.show()
