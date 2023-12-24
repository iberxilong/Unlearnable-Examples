import numpy as np
import torch
from torch import nn


def categorical_focal_loss(alpha, gamma=2.,lamda = 0.):
    """
    https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    # alpha = torch.tensor(alpha,device='cuda:0')

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = 1e-07
        y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        criterion = nn.CrossEntropyLoss()
        cross_entropy = criterion(y_pred,y_true)

        # Calculate Focal Loss
        # loss = alpha * torch.pow(1 - y_pred, gamma) * cross_entropy
        loss = alpha * torch.pow(1 - y_pred, gamma) * torch.pow(cross_entropy,lamda)

        # Compute mean loss in mini_batch
        return torch.mean(torch.sum(loss, dim=-1))

    return categorical_focal_loss_fixed