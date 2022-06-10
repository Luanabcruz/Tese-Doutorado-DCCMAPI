import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk


def find_lesion_schema(y_pred):

    def find_runs(value, a):
        # Create an array that is 1 where a is `value`, and pad each end with an extra 0.
        isvalue = np.concatenate(([0], np.equal(a, value).view(np.int8), [0]))
        absdiff = np.abs(np.diff(isvalue))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    indexes = find_runs(1, y_pred)

    # CAPTURA SOMENTE A MAIOR REGIAO CONTINUA, SE MAX_REGION == TRUE
    MAX_REGION = False
    if MAX_REGION:
        max_interval_index = None
        max_interval = -1

        for i in range(0, len(indexes)):
            diff = (indexes[i][1]-indexes[i][0])
            if max_interval < diff:
                max_interval = diff
                max_interval_index = indexes[i]

        newarr = np.zeros(len(y_pred))
        newarr[max_interval_index[0]:max_interval_index[1]] = 1

        return newarr
    else:
        newarr = np.zeros(len(y_pred))
        for i in range(0, len(indexes)):
            diff = (indexes[i][1]-indexes[i][0])
            if diff > 2:
                # max_interval = diff
                # max_interval_index = indexes[i]
                newarr[indexes[i][0]:indexes[i][1]] = 1

        return newarr


def remove_slices_no_continuos_pred(y_pred_batch):
    schema = []
    for y_pred in y_pred_batch:
        # para verificar se tem ao menos um pixel pitado na predição ou tá zerado
        if np.sum(y_pred) == 0:
            schema.append(0)
        else:
            schema.append(1)

    # print(schema)

    # print(schema)
    # print('\n')
    # aqui encontro o melhor esquema
    schema = find_lesion_schema(schema)
    # print('------------------')
    # print(schema)
    # exit(0)

    for (i, val_schema) in enumerate(schema):
        if val_schema == 0:
            y_pred_batch[i] = np.zeros(
                [y_pred_batch[i].shape[0], y_pred_batch[i].shape[1]], dtype=np.uint8)

    return y_pred_batch


def remove_slices_no_continuos_pred_by_kidney(y_pred_batch, image_list):
    schema = []
    for y_pred in y_pred_batch:
        # para verificar se tem ao menos um pixel pitado na predição ou tá zerado
        if np.sum(y_pred) == 0:
            schema.append(0)
        else:
            schema.append(1)
    # print(schema)
    schema = find_lesion_schema(schema)
    # print('aki')
    # print(schema)
    # exit()
    for (i, val_schema) in enumerate(schema):
        if val_schema == 0:
            y_pred_batch[i] = np.zeros(
                [y_pred_batch[i].shape[0], y_pred_batch[i].shape[1]], dtype=np.uint8)

    def sort_list(list1, list2):

        zipped_pairs = zip(list2, list1)

        z = [x for _, x in sorted(zipped_pairs, reverse=True)]

        return z

    tops_bb = [None] * len(schema)
    bottoms_bb = [None] * len(schema)

    for (i, val_schema) in enumerate(schema):
        if val_schema == 1:
            img = image_list[i].squeeze(
                0).transpose(1, 2, 0).astype(np.uint8).copy()
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # threshold image
            ret, threshed_img = cv2.threshold(imgray,
                                              0, 127, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            threshed_img = cv2.morphologyEx(
                threshed_img, cv2.MORPH_CLOSE, kernel)

            _, contours, hier = cv2.findContours(
                threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            bbs = []
            areas = []
            for c in contours:

                x, y, w, h = cv2.boundingRect(c)

                area = w * h

                areas.append(area)
                bbs.append((x, y, w, h))

            major_bbs = sort_list(bbs, areas)[:2]

            # if len(major_bbs) == 1:
            #     major_bbs.append(None)

            # encontra as bbs com maior área
            for bb in major_bbs:
                (x, y, w, h) = bb
                kidney_pos = 'top' if (y+w) < 128 else 'bottom'
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, kidney_pos, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

                if kidney_pos == 'top':
                    tops_bb[i] = (x, y, w, h)
                else:
                    bottoms_bb[i] = (x, y, w, h)

            # print('coutourns ', len(major_bbs))

            #     area = w * h

            #     areas.append(area)
            #     bbs.append((x, y, w, h))

            # major_bbs = sort_list(bbs, areas)[:2]

            # if len(major_bbs) == 1:
            #     major_bbs.append(None)

            # encontra as bbs com maior área

            # for bb in major_bbs:
            #     if bb != None:
            #         (x, y, w, h) = bb
            #         # print(y)
            #         kidney_pos = 'top' if (y+w) < 128 else 'bottom'

            #         if kidney_pos == 'top':
            #             tops_bb.append(bb)
            #         else:
            #             bottoms_bb.append(bb)
            #         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #         cv2.putText(img, kidney_pos, (x, y-10),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            # cv2.imshow('as', img)
            # cv2.waitKey(0)

            # fig, ax = plt.subplots(1, 2)
            # fig.canvas.set_window_title('alo')
            # ax[0].imshow(threshed_img)
            # ax[0].axis('off')
            # ax[1].imshow(img)
            # ax[1].axis('off')

            # plt.axis('off')
            # plt.show()

            # else:
            #     tops_bb.append(None)
            #     bottoms_bb.append(None)
            # exit()
    tops_schema = [0] * len(schema)
    bottoms_schema = [0] * len(schema)
    # Pega schema atual de lesões top e bottom
    for (key, y_pred) in enumerate(y_pred_batch):

        if tops_bb[key] != None:
            (x, y, w, h) = tops_bb[key]

            val = 0 if np.sum(y_pred[y:y+h, x:x+w]) == 0 else 1
            tops_schema[key] = val

        else:
            tops_schema[key] = 0

        if bottoms_bb[key] != None:
            (x, y, w, h) = bottoms_bb[key]

            val = 0 if np.sum(y_pred[y:y+h, x:x+w]) == 0 else 1
            bottoms_schema[key] = val

        else:
            bottoms_schema[key] = 0

    # print('Tops: ', tops_schema)
    # print('Bootom: ', bottoms_schema)

    # exit()
    # fig, ax = plt.subplots(1, 2)
    # fig.canvas.set_window_title("Tumor BB")
    # ax[0].imshow(y_pred[y:y+h, x:x+w])
    # ax[0].axis('off')
    # ax[1].imshow(y_pred)
    # ax[1].axis('off')

    # plt.axis('off')
    # plt.show()
    # print(key)
    # para verificar se tem ao menos um pixel pitado na predição ou tá zerado
    # if np.sum(y_pred[]) == 0:
    #     schema.append(0)
    # else:
    #     schema.append(1)

    # print('old top schema', tops_schema)
    tops_schema = find_lesion_schema(tops_schema)
    bottoms_schema = find_lesion_schema(bottoms_schema)
    # print('------------------------ NEW ONES --')
    # print('Tops: ', tops_schema)
    # print('Bootom: ', bottoms_schema)
    # exit(0)
    # print('top schema', tops_schema)
    for (i, val_schema) in enumerate(tops_schema):

        if val_schema == 0:
            if tops_bb[i] != None:

                y_pred = y_pred_batch[i]
                (x, y, w, h) = tops_bb[i]
                y_pred[y:y+h, x:x+w] = np.zeros(
                    [y_pred[y:y+h, x:x+w].shape[0],  y_pred[y:y+h, x:x+w].shape[1]], dtype=np.uint8)

    for (i, val_schema) in enumerate(bottoms_schema):

        if val_schema == 0:
            if bottoms_bb[i] != None:

                y_pred = y_pred_batch[i]
                (x, y, w, h) = bottoms_bb[i]
                y_pred[y:y+h, x:x+w] = np.zeros(
                    [y_pred[y:y+h, x:x+w].shape[0],  y_pred[y:y+h, x:x+w].shape[1]], dtype=np.uint8)

    # print('Tops: ', len(tops_bb))
    # print('Bottom: ', len(bottoms_bb))
    # exit(0)

    # y_pred_batch[i] = np.zeros(
    #     [y_pred_batch[i].shape[0], y_pred_batch[i].shape[1]], dtype=np.uint8)

    return y_pred_batch


# função de Lua para fazer um pós processamento analisando o caso em 3D
def post_processing_3D(gt_array):

    gt_array_rins = np.zeros(gt_array.shape)
    gt_array_rins[np.where(gt_array != 0)] = 1
    gt_array_rins = gt_array_rins.astype(np.uint8)

    gt_rins = sitk.GetImageFromArray(gt_array_rins)
    gt_transpose = sitk.GetImageFromArray(gt_array)  # classes originais

    stats = sitk.LabelIntensityStatisticsImageFilter()
    cc_intensity = sitk.ConnectedComponent(gt_rins)
    stats.Execute(cc_intensity, gt_rins)

    gt_original_array = np.zeros(gt_array.shape)
    volume_original_array = np.zeros(gt_array.shape)

    for ps_label in stats.GetLabels():

        # if(ps_label == None):
        # 	continue

        array_cc = sitk.GetArrayFromImage(cc_intensity)
        array_cc = ((array_cc == ps_label)*1)

        array_cc = sitk.GetImageFromArray(array_cc)

        stats = sitk.LabelStatisticsImageFilter()
        cc = sitk.ConnectedComponent(array_cc)
        stats.Execute(cc, array_cc)

        (min_x, max_x, min_y, max_y, min_z, max_z) = stats.GetBoundingBox(1)

        if ((max_z-min_z) > 1):
            max_x = max_x+2
            max_y = max_y+2
            max_z = max_z+1

            # GT
            bounding_gt = gt_transpose[min_x:max_x, min_y:max_y, min_z:max_z]
            bounding_gt_array = sitk.GetArrayFromImage(bounding_gt)
            gt_original_array[min_z:max_z, min_y:max_y,
                              min_x:max_x] = bounding_gt_array

    return gt_original_array


# função de Lua para fazer um pós processamento fechamento 3D
def post_processing_3D_morphological_closing(pred_image, kernel_value=1):

    pred_image = pred_image.astype(np.uint8)
    pred_image = sitk.GetImageFromArray(pred_image)

    morphological_c = sitk.BinaryMorphologicalClosingImageFilter()
    morphological_c.SetForegroundValue(1)
    morphological_c.SetKernelRadius(kernel_value)
    morphological_c.SetKernelType(2)
    morphological = morphological_c.Execute(pred_image)
    morphological = sitk.GetArrayFromImage(morphological)

    return morphological


def post_processing_3d_two_biggest_elem(pred_image):

    X_pred = pred_image
    X_pred = X_pred.astype(np.uint8)
    X_pred = sitk.GetImageFromArray(X_pred)

    stats = sitk.LabelIntensityStatisticsImageFilter()
    cc = sitk.ConnectedComponent(X_pred)

    stats.Execute(cc, X_pred)

    primeiro_label = None
    segundo_label = None
    size_primeiro_label = 0
    size_segundo_label = 0

    for l in stats.GetLabels():
        if (stats.GetNumberOfPixels(l) > size_primeiro_label):
            if(size_primeiro_label > size_segundo_label):
                size_segundo_label = size_primeiro_label
                segundo_label = primeiro_label
            size_primeiro_label = stats.GetNumberOfPixels(l)
            primeiro_label = l
        elif (stats.GetNumberOfPixels(l) > size_segundo_label):
            size_segundo_label = stats.GetNumberOfPixels(l)
            segundo_label = l

    # print(stats.GetNumberOfPixels(primeiro_label))
    # print(stats.GetNumberOfPixels(segundo_label))

    array_cc = sitk.GetArrayFromImage(cc)
    array_cc = ((array_cc == primeiro_label)*1) + \
        ((array_cc == segundo_label)*1)

    return array_cc
