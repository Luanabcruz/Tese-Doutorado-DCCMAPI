import numpy as np
import os
import SimpleITK as sitk
import framework.utils as ut

class CTSaver:

    def __init__(self, outdir, inlabel = 1, outlabel = 1):
        self.__outdir = outdir
        self.__inlabel = inlabel
        self.__outlabel = outlabel

    def save(self, case_name, volume):
        
        case_prediction_name = 'prediction_{}.nii.gz'.format(
            case_name.replace('case_', ''))

        output_dir = self.__outdir
        ut.create_nested_dir(output_dir)

        volume = np.where(volume == self.__inlabel, self.__outlabel, 0)

        volume = volume.transpose(1, 2, 0)

        sitk_volume = sitk.GetImageFromArray(volume)
        sitk.WriteImage(sitk_volume, os.path.join(
            output_dir, case_prediction_name))    