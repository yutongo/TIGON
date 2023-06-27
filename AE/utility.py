import numpy as np
import torch.nn as nn
import torch
import torch.nn.init as init
from sklearn.metrics import pairwise_distances
def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def compute_distance_matrix(x,p=2):
    if isinstance(x,torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
    elif isinstance(x,np.ndarray):
        distances = pairwise_distances(x)
    else:
        raise NotImplementedError
    return distances


def max_pairwise_distance(x,batch_size=1000):
    n = len(x)
    max_distance = -np.inf
    for i in range(0, n, batch_size):
        if i+batch_size<n:
            batch = x[i:i + batch_size]
        else:
            batch = x[i:]
        # distances = np.linalg.norm(batch[:, None] - x, axis=2)
        distances = pairwise_distances(batch,x[i:])
        # np.fill_diagonal(distances, -np.inf)  # Set diagonal to -inf to exclude self-distances
        max_distance = max(max_distance, np.max(distances))
    return max_distance


def init_weights_xavier_uniform(module):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)



