from sklearn.metrics import confusion_matrix
import numpy as np


def single_dice_coef(y_pred, y_true):
    # shape of y_true and y_pred: (height, width)
    intersection = np.sum(y_true * y_pred)
    if (np.sum(y_true) == 0) and (np.sum(y_pred) == 0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred))


def mean_dice_coef(y_pred, y_true):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]
    mean_dice_channel = 0.
    for i in range(batch_size):
        channel_dice = single_dice_coef(y_pred[i, :, :], y_true[i, :, :])
        mean_dice_channel += channel_dice/(batch_size)
    return mean_dice_channel


def mean_dice_coef_remove_empty(y_pred, y_true):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]
    mean_dice_channel = 0.
    num_no_empty = batch_size
    for i in range(batch_size):
        # if (np.sum(y_pred[i, :, :]) == 0) and (np.sum(y_true[i, :, :]) == 0):
        #     num_no_empty -= 1
        #     continue
        # pula todas as fatias que não possuem lesão
        if (np.sum(y_true[i, :, :]) == 0):
            num_no_empty -= 1
            continue

        channel_dice = single_dice_coef(y_pred[i, :, :], y_true[i, :, :])
        mean_dice_channel += channel_dice
    # caso o batch fica vazio retorna None para evitar que esse batch seja contabilizado
    if num_no_empty == 0:
        return None

    return mean_dice_channel/(num_no_empty)


def dice_by_case(y_pred, y_true):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]

    y_pred_new = []
    y_true_new = []

    for i in range(batch_size):
        if (np.sum(y_pred[i, :, :]) == 0) and (np.sum(y_true[i, :, :]) == 0):
            continue

        y_pred_new.append(y_pred[i, :, :])
        y_true_new.append(y_true[i, :, :])

    return dice_metric(np.asarray(y_true_new), np.asarray(y_pred_new))


def calc_metric_dict(y_true, y_pred):
    dice, jaccard, sensitivity, specificity, accuracy, prec, auc = calc_metric(
        y_true, y_pred)

    metrics = {
        'dice': dice,
        'jaccard': jaccard,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'prec': prec,
        'auc': auc
    }

    return metrics


def calc_metric(y_true, y_pred):

    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)
    except:
        dice = 0
    try:
        jaccard = (1.0 * tp) / (tp + fp + fn)
    except:
        jaccard = 0
    try:
        sensitivity = (1.0 * tp) / (tp + fn)
    except:
        sensitivity = 0
    try:
        specificity = (1.0 * tn) / (tn + fp)
    except:
        specificity = 0

    try:
        accuracy = (1.0 * (tn + tp)) / (tn + fp + tp + fn)
    except:
        accuracy = 0
    try:
        auc = 1 - 0.5 * (((1.0 * fp) / (fp + tn)) + ((1.0 * fn) / (fn + tp)))
    except:
        auc = 0
    try:
        prec = float(tp)/float(tp + fp)
    except:
        prec = 0
    #fscore = float(2*tp)/float(2*tp + fp + fn)
    return dice, jaccard, sensitivity, specificity, accuracy, prec, auc


def calc_matric_in_list(y_true_all, y_pred_all):
    metrics = []
    for i in range(0, len(y_true_all)):
        dice, jaccard, sensitivity, specificity, accuracy, auc = calc_metric(
            y_true_all[i], y_pred_all[i])
        metric = {
            'dice': dice,
            'jaccard': jaccard,
            'sensitivity': sensitivity,
            ' specificity': specificity,
            'accuracy': accuracy,
            'auc': auc
        }

        metrics.append(metric)

    return np.asarray(metrics)


def dice_metric(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    tn, fp, fn, tp = cm.ravel()
    dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)
    return dice


def jaccard_metric(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    tn, fp, fn, tp = cm.ravel()
    jaccard = (1.0 * tp) / (tp + fp + fn)
    return jaccard
