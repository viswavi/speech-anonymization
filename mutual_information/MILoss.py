import torch
import numpy as np
from .GroupSamplingMI import *


"""
expecting class-balanced training minibatch
"""
class MILoss(nn.Module):
    """
    X: discrete variable
    y: continuous variable
    """
    def forward(self, X, y, batch, n_classes=2):
        batch_size = batch.shape[0]
        group_sampling_MI = GroupSamplingMI(n_samples=batch_size, n_classes=n_classes)
        mi_loss, mi_mean, mi_stdd = group_sampling_MI(X, y, batch)
        return mi_loss