import matplotlib.pyplot as plt
import os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

channels = 3
case = "case_00184"
# outdir = r"D:\DOUTORADO_LUANA\test_kits19\npy"
# outdir = r"KiTS19_master_npy"
# outdir = r"D:\DOUTORADO_LUANA\test_kits19\npy"
# outdir = r"D:\DOUTORADO_LUANA\bases25D\KiTS19_master_npz"
outdir = r'C:\Users\usuario\Documents\UFMA\DoutoradoLuana\segmentacao2d\datasets\KiTS19_master_npz'

# dtype = "imaging"
dtype = "Images"
# dtype = "Masks"

for fname in os.listdir(os.path.join(outdir, case, dtype)):
    vol = np.load(os.path.join(outdir, case, dtype, fname))
    vol = vol['arr_0']
    # exit()
    # print(vol.shape)
    # exit()
    gt = np.load(os.path.join(outdir, case, "Masks",  "masc_"+fname))
    gt = gt['arr_0']

    if np.sum(gt) == 0:
        continue

    orig = vol

    # plt.imshow(orig, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # plt.savefig("out//"+"orig_"+fname.replace('.npz', '.png'),
    #             transparent=True, bbox_inches='tight', pad_inches=0)

    augs_dict = {
        'flip': iaa.Fliplr(1),
        'rotate': iaa.Affine(rotate=-15),
        'scale': iaa.Affine(scale=0.5),
        'pieceWise': iaa.PiecewiseAffine(scale=0.05),
        'blur': iaa.GaussianBlur(1.5),
        'contrast': iaa.LinearContrast(1.6),
        'flip_rotate': (iaa.Fliplr(1), iaa.Affine(rotate=-15)),
        'scale_pw': (iaa.Affine(scale=0.5), iaa.PiecewiseAffine(scale=0.05)),
        'blue_rotate': (iaa.GaussianBlur(1.5), iaa.Affine(rotate=-15))
    }

    for key in augs_dict:
        seq = iaa.Sequential(augs_dict[key])
        images_aug = seq(images=orig)
        plt.axis('off')
        # print(orig.shape)
        # exit()
        fig, ax = plt.subplots(2, images_aug.shape[0])

        for i in range(0, images_aug.shape[0]):
            ax[0][i].imshow(images_aug[i], cmap='gray')
            ax[0][i].axis('off')

        for i in range(0, orig.shape[0]):
            ax[1][i].imshow(orig[i], cmap='gray')
            ax[1][i].axis('off')
        plt.show()
        exit()
        # plt.imshow(images_aug, cmap='gray')
        # plt.savefig("out//"+key+"_"+fname.replace('.npz', '.png'),
        #             transparent=True, bbox_inches='tight', pad_inches=0)

    exit()
