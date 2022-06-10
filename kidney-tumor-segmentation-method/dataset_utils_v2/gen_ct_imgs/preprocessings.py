import SimpleITK as sitk
from methods.filters import histogram_matching


def preprocessing_spec_hist(image_array):
    image = sitk.GetImageFromArray(image_array)
    path = 'imaging_7.nii.gz'
    template = sitk.ReadImage(path)
    npy = histogram_matching(image, template)

    return sitk.GetArrayFromImage(npy)
