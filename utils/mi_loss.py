import torch
import numpy as np
from .GroupSamplingMI import *

"""
expecting class-balanced training minibatch
"""
class MILoss(nn.Module):
    """
    X: continuous variable
    y: discrete variable / classes
    samples_set_per_batch: hyperparameter - ideally a value smaller than the size of minibatch
    """
    def forward(self, X, y, batch, batch_size,  n_classes=2, samples_set_per_batch=1):
        group_sampling_MI = GroupSamplingMI(n_samples=batch_size//samples_set_per_batch, n_classes=n_classes)
        mi_loss, mi_mean, mi_stdd = group_sampling_MI(X, y, batch)
        return mi_loss