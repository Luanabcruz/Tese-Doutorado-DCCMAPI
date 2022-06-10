import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from scipy.stats import ttest_ind

#https://mlnotebook.github.io/post/surface-distance-function/

from scipy.ndimage import morphology

def surfd(input1, input2, sampling=1, connectivity=1):
    
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
    

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = (input_1.astype(np.float32)  - morphology.binary_erosion(input_1, conn).astype(np.float32)).astype(np.bool)
    Sprime = (input_2.astype(np.float32) - morphology.binary_erosion(input_2, conn).astype(np.float32)).astype(np.bool)

    
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
       
    
    return sds

def hausdorff(input1, input2, sampling=1, connectivity=1):
    surface_distance = surfd(input1, input2, sampling, connectivity)
    return surface_distance.max()

def average_surface_distance(input1, input2, sampling=1, connectivity=1):
    surface_distance = surfd(input1, input2, sampling, connectivity)
    return surface_distance.mean()


def metric_by_case(y_true, y_pred, metric_func):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]

    y_pred_new = []
    y_true_new = []

    for i in range(batch_size):
        if (np.sum(y_pred[i, :, :]) == 0) and (np.sum(y_true[i, :, :]) == 0):
            continue

        y_pred_new.append(y_pred[i, :, :])
        y_true_new.append(y_true[i, :, :])

    return metric_func(np.asarray(y_true_new), np.asarray(y_pred_new))


def metric_by_case_spacing(y_true, y_pred, metric_func, spacing = 1):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]

    y_pred_new = []
    y_true_new = []

    for i in range(batch_size):
        if (np.sum(y_pred[i, :, :]) == 0) and (np.sum(y_true[i, :, :]) == 0):
            continue

        y_pred_new.append(y_pred[i, :, :])
        y_true_new.append(y_true[i, :, :])

    return metric_func(np.asarray(y_true_new), np.asarray(y_pred_new), spacing)


def metric_by_slice(y_true, y_pred, metric_func):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]

    values = []

    for i in range(batch_size):
        if (np.sum(y_pred[i, :, :]) == 0) and (np.sum(y_true[i, :, :]) == 0):
            continue
        
        val = metric_func(np.asarray(y_true[i, :, :]), np.asarray(y_pred[i, :, :]))
        # if val != float("inf"):
        values.append(val)

    return np.asarray(values).mean()

def metric_by_slice_cm(y_true, y_pred, metric_func, cm):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]

    values = []

    for i in range(batch_size):
        if (np.sum(y_pred[i, :, :]) == 0) and (np.sum(y_true[i, :, :]) == 0):
            continue
        
        val = metric_func(cm)
        values.append(val)

    return np.asarray(values).mean()    

def dice(y_true, y_pred):
    try:
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
        tn, fp, fn, tp = cm.ravel()        
        dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)
    except:
        dice = 0
    return dice

def dice_cm(cm):
    try:
        tn, fp, fn, tp = cm.ravel()
        dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)
    except:
        dice = 0
    return dice    

def jaccard(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    tn, fp, fn, tp = cm.ravel()
    return (1.0 * tp) / (tp + fp + fn)

def jaccard_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return (1.0 * tp) / (tp + fp + fn)

def sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    tn, fp, fn, tp = cm.ravel()
    return (1.0 * tp) / (tp + fn)

def sensitivity_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return (1.0 * tp) / (tp + fn)


def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    tn, fp, fn, tp = cm.ravel()
    return (1.0 * tn) / (tn + fp)

def specificity_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return (1.0 * tn) / (tn + fp)    


def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    tn, fp, fn, tp = cm.ravel()
    return (1.0 * (tn + tp)) / (tn + fp + tp + fn)

def accuracy_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return (1.0 * (tn + tp)) / (tn + fp + tp + fn)


def precision(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    tn, fp, fn, tp = cm.ravel()
    return float(tp)/float(tp + fp)

def precision_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return float(tp)/float(tp + fp)


def auc(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    tn, fp, fn, tp = cm.ravel()
    return 1 - 0.5 * (((1.0 * fp) / (fp + tn)) + ((1.0 * fn) / (fn + tp)))

def auc_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return 1 - 0.5 * (((1.0 * fp) / (fp + tn)) + ((1.0 * fn) / (fn + tp)))    



def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true.flatten(), y_pred.flatten())



def pvalue(y_true, y_pred):
    return ttest_ind(y_true.flatten(), y_pred.flatten()).pvalue
