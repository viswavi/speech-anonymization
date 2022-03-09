import tqdm

import numpy as np
import pandas as pd

from .ClusterMI import *


"""
A wrapper class that performs several iterations over different sets of samples 
"""
class GroupSampler(object):

    def __init__(self, groups, sample_size):
        self.groups            = groups
        self.sample_size       = sample_size
        self.create_groups()

    def sample_groups(self):
        samples = []
        for g in self.group_samples.keys():
            samples_ = np.random.choice(self.samples_per_group[g], self.sample_size, replace=False)
            samples.extend(self.group_samples[g][samples_])
        return samples

    def create_groups(self):

        # Create dataframe
        df = pd.DataFrame(self.groups.cpu(), columns=['groups'])
        grouped = df.groupby('groups')

        # Get list with list of samples per group | Keys correspond to group labels
        groups_dict = grouped.groups    
        self.group_samples = {key:np.array(list(value)) for key, value in groups_dict.items()}

        # Get number of samples per group
        self.samples_per_group = {key:len(value) for key, value in groups_dict.items()}

class GroupSamplingMI(ClusterMI):

    def __init__(self, n_samples=1, n_classes=2, n_iterations=100, k=3, dist_metric=cosine_distance_2d):
        super(GroupSamplingMI, self).__init__(n_classes, k, dist_metric)

        self.n_samples = n_samples
        self.n_iterations = n_iterations

    def forward(self, X, y, groups):
        group_sampler = GroupSampler(groups, self.n_samples)
        
        mi = []
        for i in tqdm.tqdm(range(0, self.n_iterations)):
            idx = group_sampler.sample_groups()
            X_ = X[idx]
            y_ = y[idx]
            mi.append(self._mutual_information(X_, y_))

        mi_mean = torch.mean(torch.FloatTensor(mi))
        mi_stdd = torch.std(torch.FloatTensor(mi))
        return mi, mi_mean, mi_stdd
