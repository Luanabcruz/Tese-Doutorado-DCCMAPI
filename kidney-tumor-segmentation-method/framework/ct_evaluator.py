import framework.metrics as m
from skimage.metrics import hausdorff_distance
from sklearn.metrics import confusion_matrix
import torch
# weird trick with bincount
def confusion_matrix_2(y_true, y_pred):
    N = max(max(y_true), max(y_pred)) + 1
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat(y, torch.zeros(N * N - len(y), dtype=torch.long))
    y = y.reshape(N, N)
    return y

class CTEvaluator:

    __dict_metrics = {
        'dice': m.dice_cm,
        'iou': m.jaccard_cm,
        'sensitivity': m.sensitivity_cm,
        'specificity': m.specificity_cm,
        'accuracy': m.accuracy_cm,
        'precision': m.precision_cm,
        'auc': m.auc_cm,
        'hausdorff': m.hausdorff,
        'average_surface_distance':  m.average_surface_distance
    }

    def get_metrics_name(self):
        return self.__metrics_name

    def __init__(self, metrics_name=['dice']):
        assert set(metrics_name).issubset(
            self.__dict_metrics.keys()), "there is an invalid metric"

        self.__metrics_name = metrics_name

    def execute(self, y_true, y_pred, spacing = 1):
        try:
            cm = confusion_matrix_2(y_true.flatten(), y_pred.flatten())
        except:
            cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
            
        values = {}
        for metric_name in self.__metrics_name:
            func = self.__dict_metrics[metric_name]
            
            try:
                if metric_name == 'hausdorff' or metric_name == 'average_surface_distance':
                    value  =  m.metric_by_case_spacing(y_true, y_pred,func, spacing)
                else:
                    value = m.metric_by_slice_cm(y_true, y_pred,func,cm)
            except:
                value = 0

            values[metric_name] = str(value)

        return values
