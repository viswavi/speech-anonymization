__author__ = "Francisco Teixeira"

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor

def cosine_distance_2d(x,y):
    return 1 - F.cosine_similarity(x, y, dim=1)

def _pairwise_dists(x, dist_fn, fill_diagonal=False, diag_value=10e6):

    # Get number of elements in the vector
    N = x.shape[0]

    # Compute number of necessary rotations
    rot = int(floor(N/2))

    # Is N odd or even?
    rem = N % 2

    # Indices vector
    idx = torch.ones(N).long().to(x.device)
    idx[0] = 0
    idx = torch.cumsum(idx,0)

     # Initial x_rot corresponds to x
    x_rot = x
    idx_rot = idx

    # Init dists matrix || could probably be made more efficient in terms of memory
    # (e.g. torch.triu)
    if fill_diagonal:
        dists = diag_value*torch.ones(N,N).to(x.device)
    else:
        dists = torch.zeros(N,N).to(x.device)

    # This cycle can be parallelized
    for i in range(0, rot):

        # Rotate matrix
        x_rot = torch.roll(x_rot, 1, dims=0)
        idx_rot = torch.roll(idx_rot, 1, dims=0) # Could be done offline,
                                                 # but computational cost
                                                 # is most likely negligible

        # If N is even, the last rotation only
        # requires computing half of the distances.
        # Discard the remaining elements.
        if i == (rot-1) and rem == 0:
            x     = x[0:int(N/2)]
            x_rot = x_rot[0:int(N/2)]

            idx     = idx[0:int(N/2)]
            idx_rot = idx_rot[0:int(N/2)]

        # Compute dists
        dists_ = dist_fn(x, x_rot)

        # Save to matrix
        dists[idx, idx_rot] = dists_
        dists[idx_rot, idx] = dists_ # Matrix is symmetric,
                                     # Main diagonal is filled with zeros dist(x,x) = 0
    return dists

class ClusterMI(nn.Module):

    """
    Implementation of Nearest Neighbors approach to the computation of the MI information between a discrete and continuous datasets.
    Based on the method described in: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable

    """

    def __init__(self, n_classes=2, k=3, dist_metric=cosine_distance_2d):
        super(ClusterMI, self).__init__()

        assert n_classes >= 2, "Number of classes. Needs to be larger than or equal to two - " + str(n_classes) + " given."

        self.k = k
        self.k_digamma = torch.digamma(torch.tensor(k).float())
        self.n_classes = n_classes
        self.distance  = dist_metric

    def forward(self, X, y):
        return self._mutual_information(X,y)

    def _mutual_information(self, X, y):

        device = X.device
        N = X.shape[0] # Total number of samples
        N_digamma = torch.digamma(torch.tensor(N).float())

        # Number of samples per class
        N_x = torch.FloatTensor([torch.sum(y == i) for i in range(0, self.n_classes)])

        # Compute average digamma(N_x)
        N_x_w = N_x / N
        N_x_digammas = torch.digamma(N_x)
        avg_N_x = torch.sum(N_x_w * N_x_digammas)

        # Compute pairwise distances between all vectors in matrix X (assumed to be 2d)
        dists_matrix = _pairwise_dists(X, self.distance) #, fill_diagonal=True, diag_value=10e6)

        # Broadcast y
        y_mat = y.repeat(N,1).T # N x N matrix

        # Get same class anchor distance for each sample
        y_same_class = y_mat == y
        dists_same_class = torch.where(y_same_class, dists_matrix, 10e6*torch.ones_like(dists_matrix).to(device))  # Distances that don't matter (i.e. from another class or in the diagonal) should be very high
        anchor_dists, anchor_idx = torch.topk(dists_same_class, self.k+1, dim=1, largest=False) 
        anchor_dists = anchor_dists[:,-1]                                                       # Last dist will be the kth distance

        # Count number of samples with distance less than anchor - weird behavior
        m_i = torch.sum(torch.le(dists_matrix, anchor_dists.unsqueeze(dim=1)), dim=1) - 1  # All that are bellow anchor dist except the original value (d=0.0)
        m_i_digamma = torch.digamma(m_i.float())
        avg_m_i = torch.mean(m_i_digamma)

        # Final sum
        mutual_information = N_digamma - avg_N_x + self.k_digamma - avg_m_i
        return mutual_information / torch.log(torch.tensor(2.0))

