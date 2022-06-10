import SimpleITK as sitk
import numpy as np
import cv2
import glob

def delimitar_rins_imagem_grande(imagem_array, gt_array):

	gt_array_rins=np.zeros(gt_array.shape)
	gt_array_rins[np.where(gt_array != 0)] = 1
	gt_array_rins = gt_array_rins.astype(np.uint8)

	gt_rins = sitk.GetImageFromArray(gt_array_rins)
	
	volume_transpose = sitk.GetImageFromArray(imagem_array)
	gt_transpose = sitk.GetImageFromArray(gt_array) #classes originais
	
	stats = sitk.LabelIntensityStatisticsImageFilter()
	cc_intensity = sitk.ConnectedComponent(gt_rins)
	stats.Execute(cc_intensity,gt_rins)

	primeiro_label=None
	segundo_label=None
	size_primeiro_label=0
	size_segundo_label=0

	print(stats.GetLabels())

	for l in stats.GetLabels():
		if (stats.GetNumberOfPixels(l)>size_primeiro_label):
			if(size_primeiro_label>size_segundo_label):
				size_segundo_label=size_primeiro_label
				segundo_label=primeiro_label	
			size_primeiro_label = stats.GetNumberOfPixels(l)
			primeiro_label = l
		elif (stats.GetNumberOfPixels(l)>size_segundo_label):
			size_segundo_label = stats.GetNumberOfPixels(l)
			segundo_label = l

	label = primeiro_label,segundo_label
	gt_original_array=np.zeros(gt_array.shape)
	volume_original_array=np.zeros(gt_array.shape)

	for ps_label in label:

		if(ps_label == None):
			continue
		
		array_cc = sitk.GetArrayFromImage(cc_intensity)
		array_cc = ((array_cc==ps_label)*1)

		array_cc = sitk.GetImageFromArray(array_cc)

		stats = sitk.LabelStatisticsImageFilter()
		cc = sitk.ConnectedComponent(array_cc)
		stats.Execute(cc,array_cc)

		(min_x,max_x,min_y,max_y,min_z,max_z) = stats.GetBoundingBox(1)

		#normal
		max_x = max_x+2
		max_y = max_y+2
		max_z = max_z+1

		# porcentagem = 0.01
		# porc_min_max_x = int((max_x - min_x)*porcentagem)
		# porc_min_max_y = int((max_y - min_y)*porcentagem)
		# porc_min_max_z = int((max_z - min_z)*porcentagem)

		# min_x = min_x-porc_min_max_x
		# max_x = max_x+porc_min_max_x
		# min_y = min_y-porc_min_max_y
		# max_y = max_y+porc_min_max_y
		# min_z = min_z-porc_min_max_z
		# max_z = max_z+porc_min_max_z

		#GT
		bounding_gt = gt_transpose[min_x:max_x,min_y:max_y,min_z:max_z]
		bounding_gt_array = sitk.GetArrayFromImage(bounding_gt)
		gt_original_array[min_z:max_z,min_y:max_y,min_x:max_x] = bounding_gt_array
		
		#Volume
		bounding_vl = volume_transpose[min_x:max_x,min_y:max_y,min_z:max_z]
		bounding_vl_array = sitk.GetArrayFromImage(bounding_vl)
		volume_original_array[min_z:max_z,min_y:max_y,min_x:max_x] = bounding_vl_array

	#gt_original_array = sitk.GetImageFromArray(gt_original_array)
	#volume_original_array = sitk.GetImageFromArray(volume_original_array)

	#sitk.WriteImage(gt_original_array, "GT.nii")
	#sitk.WriteImage(volume_original_array, "IMAGEM.nii")

	return volume_original_array, gt_original_array
	#delimitar_rins_juntos2(gt_original_array, volume_original_array)

def delimitar_rins_com_textura(imagem_array, gt_array):

	gt_array_rins=np.zeros(gt_array.shape)
	gt_array_rins[np.where(gt_array != 0)] = 1
	gt_array_rins = gt_array_rins.astype(np.uint8)

	gt_rins = sitk.GetImageFromArray(gt_array_rins)
	
	volume_transpose = sitk.GetImageFromArray(imagem_array)
	gt_transpose = sitk.GetImageFromArray(gt_array) #classes originais
	
	stats = sitk.LabelStatisticsImageFilter()
	cc_intensity = sitk.ConnectedComponent(gt_rins)
	stats.Execute(cc_intensity,gt_rins)

	(min_x,max_x,min_y,max_y,min_z,max_z) = stats.GetBoundingBox(1)

	#normal
	max_x = max_x+2
	max_y = max_y+2
	max_z = max_z+1

	# porcentagem = 0.01
	# porc_min_max_x = int((max_x - min_x)*porcentagem)
	# porc_min_max_y = int((max_y - min_y)*porcentagem)
	# porc_min_max_z = int((max_z - min_z)*porcentagem)

	# min_x = min_x-porc_min_max_x
	# max_x = max_x+porc_min_max_x
	# min_y = min_y-porc_min_max_y
	# max_y = max_y+porc_min_max_y
	# min_z = min_z-porc_min_max_z
	# max_z = max_z+porc_min_max_z

	#GT
	bounding_gt = gt_transpose[min_x:max_x,min_y:max_y,min_z:max_z]
	
	#Volume
	bounding_vl = volume_transpose[min_x:max_x,min_y:max_y,min_z:max_z]

	bounding_gt = sitk.GetArrayFromImage(bounding_gt)
	bounding_vl = sitk.GetArrayFromImage(bounding_vl)

	# sitk.WriteImage(bounding_gt, "GT.nii")
	# sitk.WriteImage(bounding_vl, "IMAGEM.nii")

	return bounding_vl,bounding_gt

def function_delimited(imagem_array, gt_array):
	gt_array_rins=np.zeros(gt_array.shape)
	gt_array_rins[np.where(gt_array != 0)] = 1
	gt_array_rins = gt_array_rins.astype(np.uint8)

	gt_rins = sitk.GetImageFromArray(gt_array_rins)
	
	volume_transpose = sitk.GetImageFromArray(imagem_array)
	gt_transpose = sitk.GetImageFromArray(gt_array) #classes originais
	
	stats = sitk.LabelStatisticsImageFilter()
	cc_intensity = sitk.ConnectedComponent(gt_rins)
	stats.Execute(cc_intensity,gt_rins)

	(min_x,max_x,min_y,max_y,min_z,max_z) = stats.GetBoundingBox(1)
	#normal
	max_x = max_x+2
	max_y = max_y+2
	max_z = max_z+1

	# porcentagem = 0.05
	# porc_min_max_x = int((max_x - min_x)*porcentagem)
	# porc_min_max_y = int((max_y - min_y)*porcentagem)
	# porc_min_max_z = int((max_z - min_z)*porcentagem)

	# min_x = min_x-porc_min_max_x
	# max_x = max_x+porc_min_max_x
	# min_y = min_y-porc_min_max_y
	# max_y = max_y+porc_min_max_y
	# min_z = min_z-porc_min_max_z
	# max_z = max_z+porc_min_max_z

	positions = [min_x,max_x,min_y,max_y,min_z,max_z]

	return positions, volume_transpose, gt_transpose

def delimite_predict_origin(imagem_array, gt_array, gt_predict):

	positions_original, imagem_original, gt_original = function_delimited(imagem_array, gt_array)
	positions_prediction, imagem_prediction, gt_prediction = function_delimited(imagem_array, gt_predict)
	
	#min_x
	if(positions_prediction[0]>positions_original[0]):
		positions_prediction[0] = positions_original[0]
	else:
		positions_original[0] = positions_prediction[0]
	#max_x
	if(positions_prediction[1]<positions_original[1]):
		positions_prediction[1] = positions_original[1]
	else:
	 	positions_original[1] = positions_prediction[1]
	#min_y
	if(positions_prediction[2]>positions_original[2]):
		positions_prediction[2] = positions_original[2]
	else:
		positions_original[2] = positions_prediction[2]
	#max_y
	if(positions_prediction[3]<positions_original[3]):
		positions_prediction[3] = positions_original[3]
	else:
		positions_original[3] = positions_prediction[3]
	#min_z
	if(positions_prediction[4]>positions_original[4]):
		positions_prediction[4] = positions_original[4]
	else:
		positions_original[4] = positions_prediction[4]
	#max_z
	if(positions_prediction[5]<positions_original[5]):
		positions_prediction[5] = positions_original[5]
	else:
		positions_original[5] = positions_prediction[5]

	# GT, IMG ORIGINAL VOLUME
	gt_original_bb = gt_original[positions_original[0]:positions_original[1],positions_original[2]:positions_original[3],positions_original[4]:positions_original[5]]
	imagem_original_bb = imagem_original[positions_original[0]:positions_original[1],positions_original[2]:positions_original[3],positions_original[4]:positions_original[5]]
	
	#GT, IMG PREDICTION VOLUME
	gt_prediction_bb = gt_prediction[positions_prediction[0]:positions_prediction[1],positions_prediction[2]:positions_prediction[3],positions_prediction[4]:positions_prediction[5]]
	imagem_prediction_bb = imagem_prediction[positions_prediction[0]:positions_prediction[1],positions_prediction[2]:positions_prediction[3],positions_prediction[4]:positions_prediction[5]]

	# GT, IMG ORIGINAL ARRAY
	gt_original_bb_array = sitk.GetArrayFromImage(gt_original_bb)
	imagem_original_bb_array = sitk.GetArrayFromImage(imagem_original_bb)

	#GT, IMG PREDICTION ARRAY
	gt_prediction_bb_array = sitk.GetArrayFromImage(gt_prediction_bb)
	imagem_prediction_bb_array = sitk.GetArrayFromImage(imagem_prediction_bb)

	return imagem_original_bb_array,gt_original_bb_array,imagem_prediction_bb_array,gt_prediction_bb_array

def delimitar_rins_separadamente_original(imagem_array, gt_array):

	gt_array_rins=np.zeros(gt_array.shape)
	gt_array_rins[np.where(gt_array != 0)] = 1
	gt_array_rins = gt_array_rins.astype(np.uint8)

	gt_rins = sitk.GetImageFromArray(gt_array_rins)
	
	volume_transpose = sitk.GetImageFromArray(imagem_array)
	gt_transpose = sitk.GetImageFromArray(gt_array) #classes originais
	
	stats = sitk.LabelIntensityStatisticsImageFilter()
	cc_intensity = sitk.ConnectedComponent(gt_rins)
	stats.Execute(cc_intensity,gt_rins)

	primeiro_label=None
	segundo_label=None
	size_primeiro_label=0
	size_segundo_label=0

	for l in stats.GetLabels():
		if (stats.GetNumberOfPixels(l)>size_primeiro_label):
			if(size_primeiro_label>size_segundo_label):
				size_segundo_label=size_primeiro_label
				segundo_label=primeiro_label	
			size_primeiro_label = stats.GetNumberOfPixels(l)
			primeiro_label = l
		elif (stats.GetNumberOfPixels(l)>size_segundo_label):
			size_segundo_label = stats.GetNumberOfPixels(l)
			segundo_label = l

	label = primeiro_label,segundo_label
	gt_original_array=np.zeros(gt_array.shape)
	volume_original_array=np.zeros(gt_array.shape)

	for ps_label in label:
		
		array_cc = sitk.GetArrayFromImage(cc_intensity)
		array_cc = ((array_cc==ps_label)*1)

		array_cc = sitk.GetImageFromArray(array_cc)

		stats = sitk.LabelStatisticsImageFilter()
		cc = sitk.ConnectedComponent(array_cc)
		stats.Execute(cc,array_cc)

		(min_x,max_x,min_y,max_y,min_z,max_z) = stats.GetBoundingBox(1)

		#GT
		bounding_gt = gt_transpose[min_x:max_x,min_y:max_y,min_z:max_z]
		bounding_gt = sitk.GetArrayFromImage(bounding_gt)

		#Volume
		bounding_vl = volume_transpose[min_x:max_x,min_y:max_y,min_z:max_z]
		bounding_vl = sitk.GetArrayFromImage(bounding_vl)

		return bounding_vl,bounding_gt

def delimitar_rins_separadamente(imagem_array, gt_array, opcao_rim):
	
	min_x_esquerdo = max_x_esquerdo = min_y_esquerdo = max_y_esquerdo = min_z_esquerdo = max_z_esquerdo = 1000000
	min_x_direito = max_x_direito = min_y_direito = max_y_direito = min_z_direito = max_z_direito = -1000000
	gt_array_rins=np.zeros(gt_array.shape)
	gt_array_rins[np.where(gt_array != 0)] = 1
	gt_array_rins = gt_array_rins.astype(np.uint8)

	gt_rins = sitk.GetImageFromArray(gt_array_rins)
	
	volume_transpose = sitk.GetImageFromArray(imagem_array)
	gt_transpose = sitk.GetImageFromArray(gt_array) #classes originais
	
	stats = sitk.LabelIntensityStatisticsImageFilter()
	cc_intensity = sitk.ConnectedComponent(gt_rins)
	stats.Execute(cc_intensity,gt_rins)

	primeiro_label=None
	segundo_label=None
	size_primeiro_label=0
	size_segundo_label=0

	for l in stats.GetLabels():
		if (stats.GetNumberOfPixels(l)>size_primeiro_label):
			if(size_primeiro_label>size_segundo_label):
				size_segundo_label=size_primeiro_label
				segundo_label=primeiro_label	
			size_primeiro_label = stats.GetNumberOfPixels(l)
			primeiro_label = l
		elif (stats.GetNumberOfPixels(l)>size_segundo_label):
			size_segundo_label = stats.GetNumberOfPixels(l)
			segundo_label = l

	label = primeiro_label,segundo_label
	gt_original_array=np.zeros(gt_array.shape)
	volume_original_array=np.zeros(gt_array.shape)

	# print(label)

	for ps_label in label:

		# if(ps_label == None):
		# 	continue
		
		array_cc = sitk.GetArrayFromImage(cc_intensity)
		array_cc = ((array_cc==ps_label)*1)

		array_cc = sitk.GetImageFromArray(array_cc)

		stats = sitk.LabelStatisticsImageFilter()
		cc = sitk.ConnectedComponent(array_cc)
		stats.Execute(cc,array_cc)

		(min_x,max_x,min_y,max_y,min_z,max_z) = stats.GetBoundingBox(1)

		if(min_y < min_y_esquerdo):
			(min_x_esquerdo,max_x_esquerdo,min_y_esquerdo,max_y_esquerdo,min_z_esquerdo,max_z_esquerdo) = (min_x,max_x,min_y,max_y,min_z,max_z)
		if(min_y > min_y_direito):
			(min_x_direito,max_x_direito,min_y_direito,max_y_direito,min_z_direito,max_z_direito) = (min_x,max_x,min_y,max_y,min_z,max_z)

		#print(min_x_esquerdo,max_x_esquerdo,min_y_esquerdo,max_y_esquerdo,min_z_esquerdo,max_z_esquerdo)
		#print(min_x_direito,max_x_direito,min_y_direito,max_y_direito,min_z_direito,max_z_direito)

		#exit()

	if(opcao_rim == 1):
		(min_x,max_x,min_y,max_y,min_z,max_z) = (min_x_esquerdo,max_x_esquerdo,min_y_esquerdo,max_y_esquerdo,min_z_esquerdo,max_z_esquerdo)
	else:
		(min_x,max_x,min_y,max_y,min_z,max_z) = (min_x_direito,max_x_direito,min_y_direito,max_y_direito,min_z_direito,max_z_direito)

	max_x = max_x+2
	max_y = max_y+2
	max_z = max_z+1

	# porcentagem = 0.05
	# porc_min_max_x = int((max_x - min_x)*porcentagem)
	# porc_min_max_y = int((max_y - min_y)*porcentagem)
	# porc_min_max_z = int((max_z - min_z)*0.05)

	# min_x = min_x-porc_min_max_x
	# max_x = max_x+porc_min_max_x
	# min_y = min_y-porc_min_max_y
	# max_y = max_y+porc_min_max_y
	# min_z = min_z-porc_min_max_z
	# max_z = max_z+porc_min_max_z

	#GT
	bounding_gt = gt_transpose[min_x:max_x,min_y:max_y,min_z:max_z]
	bounding_gt = sitk.GetArrayFromImage(bounding_gt)

	#Volume
	bounding_vl = volume_transpose[min_x:max_x,min_y:max_y,min_z:max_z]
	bounding_vl = sitk.GetArrayFromImage(bounding_vl)

	return bounding_vl,bounding_gt