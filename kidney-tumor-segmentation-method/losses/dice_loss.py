import torch


# PyTorch
# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=0):

#         # comment out if your model contains a sigmoid or equivalent activation layer
#         #inputs = F.sigmoid(inputs)

#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         intersection = (inputs * targets).sum()
#         dice = (2.*intersection + smooth) / \
#             (inputs.sum() + targets.sum() + smooth)

#         return 1 - dice


class diceloss(torch.nn.Module):

    def init(self):
        super(diceLoss, self).init()

    def forward(self, pred, target):
        smooth = 1
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


if __name__ == '__main__':
    import numpy as np
    pred = np.zeros([256, 256], dtype=float)
    true = np.zeros([256, 256], dtype=float)

    pred = torch.from_numpy(pred)
    true = torch.from_numpy(true)

    criterion = diceloss()

    print('Value ', criterion(pred, true))
