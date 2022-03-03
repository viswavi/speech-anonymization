import torch
import numpy as np
from .GroupSamplingMI import *


"""
expecting class-balanced input by batch
and target/labels by batch
"""
class MILoss(object):
    def forward(self, input, target, n_classes=2):
        # balanced_input, balanced_target, batch_size = _class_balanced_sampling_by_batch(input, target)
        batch_size = input.shape[0]

        group_sampling_MI = GroupSamplingMI(n_samples=batch_size, n_classes=n_classes)

        mi_loss, mi_mean, mi_stdd = group_sampling_MI(input, target, )
        return mi_loss