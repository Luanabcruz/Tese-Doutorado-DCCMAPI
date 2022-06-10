from .metrics import calc_metric, calc_metric_dict
from .dataset import load_casebyname_from_image_folder
import numpy as np
import torch
import math
import SimpleITK as sitk
import matplotlib.pyplot as plt
from .post_process import post_processing_3D, post_processing_3D_morphological_closing, post_processing_3d_two_biggest_elem

import cv2
import os


def pred_one_slice(model, image, mask, threshold=0.5):
    with torch.no_grad():
        im = image
        mask = mask

        # transform binary mask between 0 and 1
        y_true = np.where(mask == 255, 1, 0)

        output = model(torch.from_numpy(
            im).type(torch.cuda.FloatTensor)/255)

        y_pred = output['out'].cpu().detach().numpy()

        # applies the threshold
        if threshold is None:
            y_pred = np.where(y_pred > (y_pred.max()/2), 1, 0)
        else:
            y_pred = np.where(y_pred > threshold, 1, 0)

        # remove extra dimensions
        y_pred = np.array(y_pred[-1, -1, :, :], dtype=np.uint8)

    return np.asarray(y_true), np.asarray(y_pred)


def pred_one_case(model, case,  cases_template, label=1, case_name='', threshold=0.5, only_injuries_slices=False, debug=False, post_process=False, post_process_type=None, save_pred=False):
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for i in range(0, len(case['image'])):
            im = case['image'][i]
            mask = case['mask'][i]

            # transform binary mask between 0 and 1
            # y_true = np.where(mask == 255, 1, 0)
            y_true = mask
            # remover fatias sem lesão
            if only_injuries_slices:
                if np.sum(y_true) == 0:
                    continue

            output = model(torch.from_numpy(
                im).type(torch.cuda.FloatTensor))

            # y_pred = output['out'].cpu().detach().numpy()
            y_pred = output.cpu().detach().numpy()

            # applies the threshold
            if threshold is None:
                y_pred = np.where(y_pred > (y_pred.max()/2), 1, 0)
            else:
                y_pred = np.where(y_pred > threshold, 1, 0)

            # remove extra dimensions
            y_pred = np.array(y_pred[-1, -1, :, :], dtype=np.uint8)

            if cases_template is not None:
                if int(cases_template[case_name][i]) == 0:
                    y_pred = np.zeros(
                        [im.shape[2], im.shape[3]], dtype=np.uint8)

            if debug:
                fig, ax = plt.subplots(1, 3)
                fig.canvas.set_window_title("case code: ")
                ax[0].imshow(y_pred, cmap='gray')
                ax[0].title.set_text("predicao")
                ax[0].axis('off')
                ax[1].imshow(mask, cmap='gray')
                ax[1].title.set_text("mascara")
                ax[1].axis('off')
                ax[2].imshow(im.squeeze().transpose(
                    1, 2, 0)[:, :, 1],  cmap='gray')
                ax[2].title.set_text("imagem")
                # ax[2].imshow(im.squeeze(),  cmap='gray')

                ax[2].axis('off')
                plt.axis('off')
                plt.show()

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)

    foldername = ''
    if post_process:
        # TODO: criar uma versão generica do pós que contemple tanto o pos da lesao quanto do rim. Sugiro usar o delegate para a funcao de pos

        # y_pred_all = post_processing_3d_two_biggest_elem(
        #     np.asarray(y_pred_all))

        y_pred_all = post_processing_3D(np.asarray(y_pred_all))

        # y_pred_all = post_processing_3D_morphological_closing(y_pred_all)

        # if post_process_type is None:
        #     foldername = 'full_pos'
        #     y_pred_all = post_processing_3D(np.asarray(y_pred_all))
        #     # y_pred_all = post_processing_3D_morphological_closing(
        #     #     y_pred_all, kernel_value=1)
        # elif post_process_type == 'remove_slices':
        #     y_pred_all = post_processing_3D(np.asarray(y_pred_all))
        # elif post_process_type == 'morphological_closing':
        #     y_pred_all = post_processing_3D_morphological_closing(
        #         np.asarray(y_pred_all))

        if post_process_type is not None:
            foldername = post_process_type
    else:
        foldername = 'sem_pos'

    if save_pred:
        # save_pred_img(case_name, y_pred_all)
        # save_pred_img(case_name, y_true_all, suffix='GT')
        save_pred_nii(case_name, y_pred_all, label,
                      suffix='_' + str(save_pred))
        # save_pred_nii(case_name, y_true_all, suffix='_'+'GT_v2')

    return np.asarray(y_true_all), np.asarray(y_pred_all)


def save_pred_img(case_name, y_pred_case, cases_mask_name=None, out_dir=None, suffix=''):
    if out_dir is None:
        output_dir = os.path.join(
            '.', 'output_predict_img_{}'.format(suffix), case_name)
    else:
        output_dir = os.path.join(
            out_dir, 'output_predict_img_{}'.format(suffix), case_name)

    # save pred
    try:
        os.makedirs(output_dir)
    except:
        pass

    i = 0
    for y_aux in y_pred_case:
        y_pred_image = y_aux.copy()

        y_pred_image = np.where(y_pred_image == 1, 255, 0)
        if cases_mask_name is not None:
            cv2.imwrite(os.path.join(
                output_dir, cases_mask_name[i]), y_pred_image)
        else:
            cv2.imwrite(os.path.join(
                output_dir, "pred_{}-{}.png".format(case_name, i)), y_pred_image)
        i += 1


def save_pred_nii(case_name, y_pred_case, label, out_dir=None, case_info_folder=None, suffix=''):
    case_prediction_name = 'prediction_{}.nii.gz'.format(
        case_name.replace('case_', ''))

    # case_prediction_name = 'pred_{}.nii.gz'.format(case_name)
    if out_dir is None:
        # output_dir = os.path.join('.', 'output_predict_nii', case_name)
        output_dir = os.path.join('.', 'output_predict_nii{}'.format(suffix))
    else:
        output_dir = os.path.join(
            out_dir, 'out_pred_nii{}'.format(suffix))
    # save pred
    try:
        os.makedirs(output_dir)
    except:
        pass

    y_pred_case = np.asarray(y_pred_case)

    y_pred_case = np.where(y_pred_case == 1, label, 0)

    y_pred_case = y_pred_case.transpose(1, 2, 0)

    y_pred_case = sitk.GetImageFromArray(y_pred_case)

    if case_info_folder is not None:
        nii_file_info = sitk.ReadImage(os.path.join(
            case_info_folder, case_prediction_name))
        y_pred_case.CopyInformation(nii_file_info)

    sitk.WriteImage(y_pred_case, os.path.join(
        output_dir, case_prediction_name))


def pred_all(model, test_root_data_dir, cases_codes, label=1, cases_template=None, only_injuries_slices=False, debug=False, threshold=0.5,    post_process=False, post_process_type=None, save_pred=False, image_folder='Images', mask_folder='Masks'):

    history = {}
    list_metrics = {}
    # print("\n\nPaciente   | Dice | IoU | Sen | Esp| Acc | Prec \n")
    print("\n\nPaciente   | Dice  \n")
    for case_name in cases_codes:

        case = load_casebyname_from_image_folder(
            test_root_data_dir, case_name, image_folder=image_folder, mask_folder=mask_folder)

        y_true, y_pred = pred_one_case(
            model, case, case_name=case_name, label=label, threshold=threshold, cases_template=cases_template, only_injuries_slices=only_injuries_slices, post_process=post_process, post_process_type=post_process_type, save_pred=save_pred, debug=debug)

        metrics = calc_metric_dict(y_true, y_pred)

        for m in metrics.keys():
            if not m in list_metrics:
                list_metrics[m] = []

            list_metrics[m].append(metrics[m])

        print("{} ==> {:.4f}".format(case_name, metrics['dice']))
        # print("{} => {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
        #     case_name, metrics['dice'], metrics['jaccard'], metrics['sensitivity'], metrics['specificity'], metrics['accuracy'], metrics['prec']))

        history[case_name] = {
            'dice':  metrics['dice'],
            'jaccard': 0
        }

    summary = {}
    for m in list_metrics.keys():
        list_metrics[m] = np.asarray(list_metrics[m])
        summary[m] = '{} (std: {})'.format(
            list_metrics[m].mean(), list_metrics[m].std())

        summary["excel_version_" +
                m] = '=ROUND({},4)&"±"&ROUND({},2)'.format(list_metrics[m].mean(), list_metrics[m].std())

    history['results_final'] = summary

    return history


def pred_all_by_slices(model, test_root_data_dir, cases_codes, only_injuries_slices=False, debug=False, threshold=0.5,    post_process=False, image_folder='Images', mask_folder='Masks',):
    dices = []
    jaccards = []

    history = {}

    for case_name in cases_codes:
        print("CASE ", str(case_name))

        case = load_casebyname_from_image_folder(
            test_root_data_dir, case_name, image_folder=image_folder, mask_folder=mask_folder)

        for i in range(0, len(case['image'])):

            y_true, y_pred = pred_one_slice(
                model, case['image'][i], case['mask'][i])

            dice, jaccard, _, _, _, _ = calc_metric(y_true, y_pred)

        if not math.isnan(dice):
            dices.append(dice)

        if not math.isnan(jaccard):
            jaccards.append(jaccard)

    dices = np.asarray(dices)
    jaccards = np.asarray(jaccards)
    history['results_final'] = {
        'dice': '{} (std: {})'.format(dices.mean(), dices.std()),
        'jaccard': '{} (std: {})'.format(jaccards.mean(), jaccards.std()),
        'excel_version_dice': '=ROUND({},4)&"±"&ROUND({},2)'.format(dices.mean(), dices.std()),
        'excel_version_jaccard': '=ROUND({},4)&"±"&ROUND({},2)'.format(jaccards.mean(), jaccards.std()),
    }
    return history


def pred_all_template(model, test_root_data_dir, cases_codes, image_folder='Images', mask_folder='Masks',):

    cases_template = {}

    for case_name in cases_codes:
        print("=> CASE ", str(case_name))

        case = load_casebyname_from_image_folder(
            test_root_data_dir, case_name, image_folder=image_folder, mask_folder=mask_folder)
        template = []
        for i in range(0, len(case['image'])):

            y_true, y_pred = pred_one_slice(
                model, case['image'][i], case['mask'][i])

            if np.sum(y_pred) == 0:
                y_pred = 0
            else:
                y_pred = 1

            template.append(str(y_pred))

        print(template)
        cases_template[case_name] = template

    return cases_template


def pred_one_case_and_save(model, case, label, cases_template, case_name='', threshold=0.5, post_process=False, out_dir=None, case_info_folder=None):
    y_pred_all = []
    with torch.no_grad():
        for i in range(0, len(case['image'])):
            im = case['image'][i]

            # exit()
            output = model(torch.from_numpy(im).type(torch.cuda.FloatTensor))

            # y_pred = output['out'].cpu().detach().numpy()
            y_pred = output.cpu().detach().numpy()

            # applies the threshold
            if threshold is None:
                y_pred = np.where(y_pred > (y_pred.max()/2), 1, 0)
            else:
                y_pred = np.where(y_pred > threshold, 1, 0)

            # remove extra dimensions
            y_pred = np.array(y_pred[-1, -1, :, :], dtype=np.uint8)

            # if True:
            #     fig, ax = plt.subplots(1, 3)
            #     fig.canvas.set_window_title("case code: ")
            #     ax[0].imshow(y_pred, cmap='gray')
            #     ax[0].axis('off')
            #     ax[1].imshow(y_pred, cmap='gray')
            #     ax[1].axis('off')
            #     ax[2].imshow(im.squeeze().transpose(
            #         1, 2, 0)[:, :, 1],  cmap='gray')
            #     ax[2].axis('off')
            #     plt.axis('off')
            #     plt.show()

            if cases_template is not None:
                if int(cases_template[case_name][i]) == 0:
                    y_pred = np.zeros(
                        [im.shape[2], im.shape[3]], dtype=np.uint8)

            y_pred_all.append(y_pred)

    if post_process:
        y_pred_all = post_processing_3D(
            np.asarray(y_pred_all))

        # y_pred_all = post_processing_3d_two_biggest_elem(
        #     np.asarray(y_pred_all))
        # y_pred_all = post_processing_3D_morphological_closing(y_pred_all)

    save_pred_nii(case_name, y_pred_all, label, out_dir=out_dir,
                  case_info_folder=case_info_folder)
    # save_pred_img(case_name, y_pred_all, out_dir=out_dir)


def pred_and_save(model, test_root_data_dir, cases_codes, label, cases_template=None, only_injuries_slices=False, debug=False, threshold=0.5, post_process=False, save_pred=False, image_folder='Images', mask_folder='Masks', out_dir=None, case_info_folder=None):

    for case_name in cases_codes:
        print("CASE ", str(case_name))

        case = load_casebyname_from_image_folder(
            test_root_data_dir, case_name, image_folder=image_folder, mask_folder=None)

        pred_one_case_and_save(
            model, case, case_name=case_name, label=label,  threshold=threshold, cases_template=cases_template,  post_process=post_process, out_dir=out_dir, case_info_folder=case_info_folder)
