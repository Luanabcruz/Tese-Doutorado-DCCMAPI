from dataset_loader import DatasetLoader
import numpy as np
import os
import SimpleITK as sitk
import dataset_utils as ut
from cache_management.nift_cache import NiftCache
from cache_management.general_cache import GeneralCache
from preprocessings import Preprocessing

class NiftLoader(DatasetLoader):

    def __init__(self, path: str, preprocessing: Preprocessing  = None) -> None:
        super().__init__()
        self.__path = path
        self.__cases = os.listdir(path)
        self.__data_cache = NiftCache()
        self.__general_cache = GeneralCache()
        self.__preprocessing = preprocessing
        
    def getCasesName(self) -> list:
        return self.__cases

    def getCaseDataByIndex(self, index) -> np: 
        assert index >= 0 and index <  len(self.__cases), "Indice inválido!"
        pass
    
    def getCaseDataByName(self, name) -> np:
        assert name in self.__cases, "Caso inexistente!"

        if self.__data_cache.is_file_exists(name):
            print("Loading {} from cache...".format(name))
            return self.__data_cache.load(name)
        else:
            print("processing {} ...".format(name))

        sitk_image = sitk.ReadImage(os.path.join(self.__path, name,"imaging.nii.gz"))
        np_image = sitk.GetArrayFromImage(sitk_image)

        sitk_gt = sitk.ReadImage(os.path.join(self.__path, name,"segmentation.nii.gz"))
        np_gt = sitk.GetArrayFromImage(sitk_gt)
        
        np_image = ut.windowing(np_image)
        np_image, np_gt = ut.remove_empty_slices_by_label(np_image, np_gt, label = 2)

        if self.__preprocessing is not None:
            np_image = self.__preprocessing.forward(np_image, sitk_image)
            np_gt = self.__preprocessing.forward(np_gt, sitk_gt)

        np_image = ut.rescale_intensity(np_image, rgb_scale = True)

        np_image = ut.apply_mask_on_image_by_label(np_image, np_gt, label = 2)

        self.__data_cache.save(name, np_image)

        return np_image

    def getAllsCases(self, name):
        pass

    def getMaxSlices(self):
        num_max = 256

        if self.__general_cache.is_file_exists("maxSlices"):
            return self.__general_cache.load("maxSlices")["maxSlices"]

        for i in range(len(self.__cases)):
            num_max = max(num_max, self.getCaseDataByName(self.__cases[i]).shape[2])
            
        self.__general_cache.save("maxSlices", {"maxSlices": num_max})    
        return num_max

if __name__ == '__main__':
    path = r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'
    dataset = NiftLoader(path)
    print("Max num slices", dataset.getMaxSlices())
   

