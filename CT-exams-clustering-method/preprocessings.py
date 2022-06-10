from abc import ABC, abstractmethod
import SimpleITK as sitk
import numpy as np
import os

class Preprocessing(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod    
    def forward(image: sitk.Image):
        pass



class NiftResizer(Preprocessing):
    
    def __init__(self, size):
        self._size = size
      
    def forward(self, image_np: np, reference: sitk.Image) -> sitk.Image:

        new_nii = resize(image_np, reference, self._size)
        nii_arr = sitk.GetArrayFromImage(new_nii)
        return nii_arr


def resize(image_np, reference:sitk.Image, output_size):
    image= sitk.GetImageFromArray(image_np)
    new_spacing = [old_sz*old_spc/new_sz for old_sz, old_spc, new_sz in zip(image.GetSize(), reference.GetSpacing(), output_size)]
    interpolator_type = sitk.sitkLinear
    return sitk.Resample(image, output_size, sitk.Transform(), interpolator_type, reference.GetOrigin(), new_spacing, image.GetDirection(), 0.0, reference.GetPixelIDValue())


if __name__ == '__main__':
    import os

    path = r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'
    case = "case_{:05d}"
    case_rez = case.format(42)
    case_ref = case.format(10)

    nii_rez = sitk.ReadImage(os.path.join(path, case_rez, "imaging.nii.gz"))
    print("Inicial:", nii_rez.GetSize())
   
    # new_nii = resize_sitk(nii_rez, nii_ref)
    new_nii = resize(nii_rez, (50, 512,512))

    # resizer = NiftResizer((512,512, 27))

    # nii_arr = sitk.GetArrayFromImage(nii_rez)

    # new_nii = resizer.forward(nii_arr, nii)
    print("Final:", new_nii.GetSize())
    sitk.WriteImage(new_nii,os.path.join('out', 'test', "case_174.nii.gz") )


   


# img = sitk.ReadImage("imaging.nii.gz")

# #print(img.GetSize())
	
# img_interpolada = resize_image(img, (51,128,128))