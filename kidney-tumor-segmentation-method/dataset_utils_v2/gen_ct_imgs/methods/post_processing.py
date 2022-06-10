import SimpleITK as sitk
import numpy as np

def removes_non_continuous_elements(pred_array):

	pred_array_rins=np.zeros(pred_array.shape)
	pred_array_rins[np.where(pred_array != 0)] = 1
	pred_array_rins = pred_array_rins.astype(np.uint8)

	pred_rins = sitk.GetImageFromArray(pred_array_rins)
	pred_transpose = sitk.GetImageFromArray(pred_array) #classes originais
	
	stats = sitk.LabelIntensityStatisticsImageFilter()
	cc_intensity = sitk.ConnectedComponent(pred_rins)
	stats.Execute(cc_intensity,pred_rins)

	pred_original_array=np.zeros(pred_array.shape)
	volume_original_array=np.zeros(pred_array.shape)

	for ps_label in stats.GetLabels():

		# if(ps_label == None):
		# 	continue
		
		array_cc = sitk.GetArrayFromImage(cc_intensity)
		array_cc = ((array_cc==ps_label)*1)

		array_cc = sitk.GetImageFromArray(array_cc)

		stats = sitk.LabelStatisticsImageFilter()
		cc = sitk.ConnectedComponent(array_cc)
		stats.Execute(cc,array_cc)

		(min_x,max_x,min_y,max_y,min_z,max_z) = stats.GetBoundingBox(1)

		if ((max_z-min_z) > 1):
			max_x = max_x+2
			max_y = max_y+2
			max_z = max_z+1

			#GT
			bounding_pred = pred_transpose[min_x:max_x,min_y:max_y,min_z:max_z]
			bounding_pred_array = sitk.GetArrayFromImage(bounding_pred)
			pred_original_array[min_z:max_z,min_y:max_y,min_x:max_x] = bounding_pred_array

	return pred_original_array

def morphological_closing(pred_image,kernel_value):
	morphological_c = sitk.BinaryMorphologicalClosingImageFilter()
	morphological_c.SetForegroundValue(1)
	morphological_c.SetKernelRadius(kernel_value)
	#morphological_c.SetKernelType(0)
	morphological = morphological_c.Execute(pred_image)
	morphological = sitk.GetArrayFromImage(morphological)

	return morphological