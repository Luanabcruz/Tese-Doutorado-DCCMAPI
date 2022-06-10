import torch
import os
import numpy as np
import framework.ct_dataset_manager as dm
import framework.utils as ut
from datetime import datetime
import time

import matplotlib.pyplot as plt
import SimpleITK as sitk

import pandas as pd

from dataset_utils_v2.gen_ct_imgs.gen_tumor_case_data_by_pred_kidney_25D_256 import generate_tumor_image_mask
from dataset_utils_v2.gen_ct_imgs.resize_gt_back_orig_512 import resize_back


class VolumePredictor:

    def __init__(self, model):
        self.__model = model

    """ 
    Entrada: 
        Volume Nift: (D,C,W,H) -> Depth (slices), Channels, Width, Height
    """ 
    def predict(self, volume, batch_size = 8):
        assert volume.ndim == 4, "Volume informado tem shape inválido"
        
        pred_volume = None
        
        with torch.no_grad():  
            for i in range(0, volume.shape[0], batch_size):
                image = volume[i:i+batch_size]
                image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
                out  = self.__model(image)
                out = out.squeeze(1)
                pred_vol = out.cpu().numpy()
                
                if pred_volume is None:
                    pred_volume = pred_vol
                else:
                    pred_volume = np.vstack((pred_volume, pred_vol))

        return pred_volume

class  LuaInference2D:

    __threshold = 0.5
    __nii_label = 1
    __verbose = True
    __report_filename_template = 'report_{}.csv'
    __current_filename = ''
    __report_head = ''

    
    def __init__(self, dataset_dir, model, evaluator, nii_save=False, output_dir='./out', cases_info_dir=None, prob_out = False):
        self.__dataset_dir = dataset_dir
        self.__dataset_type = "npy"
        self.__dataset_dim = "256"
        self.__model = model
        self.__output_dir = output_dir
        self.__nii_save = nii_save
        self.__evaluator = evaluator
        self.__cases_info_dir = cases_info_dir
        self.__after_init()
        self.__channels = 3
        self.__rgb = False
        self.__df = None
        self.__prob_out = prob_out
        self._execution_time = []

    def set_dataset_npy(self):
        self.__dataset_type = "npy"

    def set_dataset_cascade(self):
        self.__dataset_type = "cascade"    

    def set_dataset_dim_256(self):
        self.__dataset_dim = "256"

    def set_dataset_dim_512(self):
        self.__dataset_dim = "512"           

    def __get_case_info_image(self, case_number):

        im_path = os.path.join(self.__cases_info_dir, "case_{:05d}".format(case_number),'segmentation.nii.gz')
        im_k = sitk.ReadImage(im_path)

        return im_k


    def disable_verbose(self):
        self.__verbose = False

    def enable_verbose(self):
        self.__verbose = True    

    def set_channels(self,value):
        self.__channels = value    
    
    def set_is_rgb(self,value):
        self.__rgb = value


    def __after_init(self):

        self.__report_head = ["cases"]
        self.__report_head.extend(
            [m for m in self.__evaluator.get_metrics_name()])

        ut.create_nested_dir(self.__output_dir)

    def set_threshold(self, thresh):
        self.__threshold = thresh

    def set_output_dir(self, dirname):
        self.__output_dir = dirname

    def set_nii_save(self, value):
        self.__nii_save = value

    def set_nii_label(self, label):
        self.__nii_label = label

    def getLastExecution(self):
        return np.asarray(self._execution_time)
    
    def execute(self, cases, batch_size=8, post_processings=None):
        self._execution_time = []
        # starting time
        
        # self.__create_report_file()

        keys = ['case_name']
        keys.extend(self.__evaluator.get_metrics_name())
        
        df = pd.DataFrame(columns=keys)

        for case_name in cases:
            start = time.time()
            # realiza a inferência em um único volume CT
            y_true, y_pred = self.execute_one(case_name, batch_size)
            # print(case_name)
            # aplica os pós processamentos
            y_pred = self.__apply_post_processings(y_pred, post_processings)
                       
            if not self.__cases_info_dir is None:
                gt_sitk_image= self.__get_case_info_image(int(case_name.replace("case_","")))
                spacing= gt_sitk_image.GetSpacing()
                spacing = tuple(ti/2 for ti in spacing)
            else: 
                spacing = None
            # avalia as predições
            result = self.__evaluator.execute(y_true, y_pred, spacing)
            result["case_name"] = case_name
            
            df = df.append(result, ignore_index=True)

            # str_result = ', '.join([str(elem) for elem in result])

            if self.__verbose:
                out_str = "{}, ".format(case_name) 
                for metric in self.__evaluator.get_metrics_name():
                    out_str += "{}, ".format(result[metric])
                print(out_str)
                # print("{}, {}, {}".format(case_name, result['dice'], result['sensitivity']))
                # print("{}, {}".format(case_name, result['dice']))
            
            if self.__nii_save:
                self.__save_inference_nii(case_name, y_pred)
                # self.__save_gt_nii(case_name, y_true)
                # self.__save_imaging_nii(case_name)

            end = time.time()
        
            self._execution_time.append((end-start))
            # salva as métricas em um arquivo
            
            # self.__save_result_report(case_name, result)
        df.to_csv(os.path.join(self.__output_dir, 'result_cases.csv'), index=False, sep =';', decimal=',')
        
        self.__df = df
        # end time
        

    def get_result_dataframe(self):
        return self.__df
     
    def execute_one(self, case_name, batch_size=8):
        
        if self.__dataset_type == 'npy':
            case = dm.load_case_from_folder(self.__dataset_dir, case_name)
        else: 
            case = generate_tumor_image_mask(self.__dataset_dir, self.__cases_info_dir, case_name)

        if self.__rgb:
            case['image'] = case['image']/255
        
        y_pred_all = None
        y_true_all = case['mask']
       
        predictor = VolumePredictor(self.__model)          
        y_pred_all = predictor.predict(case['image'].squeeze(1), batch_size)

        if not self.__prob_out:
            y_pred_all = self.__apply_threshold(y_pred_all)    
           
        if self.__dataset_dim == "512":
            y_pred_all = resize_back(np.asarray(y_pred_all), case["info"])

            y_true_all =self.__get_case_info_image(int(case_name.replace("case_","")))
            y_true_all = sitk.GetArrayFromImage(y_true_all).transpose(2, 0, 1)
            y_true_all = np.where(y_true_all == 1, 0, y_true_all).astype(np.uint8)
            y_true_all = np.where(y_true_all == 2, 1, y_true_all).astype(np.uint8)
           

        return np.asarray(y_true_all), np.asarray(y_pred_all)

    def __save_inference_nii(self, case_name, y_pred):
        case_prediction_name = 'prediction_{}.nii.gz'.format(
            case_name.replace('case_', ''))
        # output_dir = os.path.join(self.__output_dir, 'nii')
        output_dir = self.__output_dir
        ut.create_nested_dir(output_dir)

        y_pred = np.where(y_pred == 1, self.__nii_label, 0)

        y_pred = y_pred.transpose(1, 2, 0)

        sitk_y_pred = sitk.GetImageFromArray(y_pred)

        # if self.__cases_info_dir is not None:
        #     nii_file_info = self.__get_case_info_image(int(case_name.replace("case_","")))
        #     sitk_y_pred.SetSpacing(nii_file_info.GetSpacing())
            # sitk_y_pred.CopyInformation(nii_file_info)

        sitk.WriteImage(sitk_y_pred, os.path.join(
            output_dir, case_prediction_name))

    def __save_gt_nii(self, case_name, y_true):
        case_seg_name = 'segmentation_{}.nii.gz'.format(
            case_name.replace('case_', ''))
        
        output_dir = os.path.join(self.__output_dir, 'nii')

        ut.create_nested_dir(output_dir)

        y_true = np.where(y_true == 1, self.__nii_label, 0)

        y_true = y_true.transpose(1, 2, 0)

        sitk_y_true = sitk.GetImageFromArray(y_true)

        # if self.__cases_info_dir is not None:
        #     nii_file_info = sitk.ReadImage(os.path.join(
        #         self.__cases_info_dir, case_seg_name))
        #     sitk_y_true.CopyInformation(nii_file_info)

        sitk.WriteImage(sitk_y_true, os.path.join(
            output_dir, case_seg_name))

    def __save_imaging_nii(self, case_name):
         
        imaging = dm.load_case_from_folder(self.__dataset_dir, case_name)['image']
        # Se faz necessário, pois está no 2.5D. E quero pegar somente o central
        imaging = imaging.squeeze()[:, 1, :]
        case_imaging_name = 'imaging.nii.gz'
        output_dir = os.path.join(self.__output_dir, 'nii',case_name)
        ut.create_nested_dir(output_dir)

        imaging = imaging.transpose(1, 2, 0)

        sitk_imaging = sitk.GetImageFromArray(imaging)

        if self.__cases_info_dir is not None:
            nii_file_info = sitk.ReadImage(os.path.join(
                self.__cases_info_dir, case_imaging_name))
            sitk_imaging.CopyInformation(nii_file_info)

        sitk.WriteImage(sitk_imaging, os.path.join(
            output_dir, case_imaging_name))        

    def __apply_post_processings(self, vol, post_processings):

        if post_processings is None or len(post_processings) == 0:
            return vol

        for post_proc in post_processings:
            vol = post_proc(vol)

        return vol

    def __predict_im(self, im):
        output = self.__model(torch.from_numpy(
            im).type(torch.cuda.FloatTensor))
        out = output.cpu().detach().numpy()
        # remoção das dimensões extras
        out = out[-1, -1, :, :]

        return self.__apply_threshold(out)

    def __apply_threshold(self, out):
        # essa era a ideia do threshold variável
        if self.__threshold is None:
            out = np.where(out > (out.max()/2), 1, 0)
        else:
            out = np.where(out > self.__threshold, 1, 0)

        return np.array(out, dtype=np.uint8)

    def __create_report_file(self):
        now = datetime.now()
        # str_datetime = now.strftime("%Y%m%d_%H_%M_%S")
        str_datetime = "_#"
        self.__current_filename = self.__report_filename_template.format(
            str_datetime)
        row = ''
        with open(os.path.join(self.__output_dir, self.__current_filename), 'w', newline='') as csvfile:
            row += ','.join(self.__report_head)
            csvfile.write(row)
            csvfile.write('\n')

    def __save_result_report(self, case_name, result):
        row = ''
        with open(os.path.join(self.__output_dir, self.__current_filename), 'a', newline='') as csvfile:
            row += 'case_name,'
            row += ','.join(result)
            csvfile.write(row)
            csvfile.write('\n')


if __name__ == '__main__':

    import torch
    import ct_evaluator as ct_eval
    import post_processings as pp
    import sys
    sys.path.append(r'E:\DoutoradoLuana\codes\luanet')

    dataset = r'E:\DoutoradoLuana\datasets\gerados\Kits19_25D_with_tumor_masks_new_norm_and_windowing'
    # weigths_path = r'../pesos_luanet/tumor/TW_V5_1_resunet101.pt'
    weigths_path = r'./logs_online_aug/unet_dice_resnet101_dist_natal_v7_ch_3_challenge/weights_partial_diceval_epch32_20210318_03_19_12.pt'

    model = torch.load(weigths_path)

    evaluator = ct_eval.CTEvaluator(['dice'])

    cases = dm.load_dataset_dist(dataset_id="natal_v7")['test']

    inference = LuaInference2D(dataset, model, evaluator, True)
    inference.set_nii_label(2)
    inference.execute(
        cases, False, [pp.min_slices_number_3D, pp.median_filter])
