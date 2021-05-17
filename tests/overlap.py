import torch
import numpy as np


def get_overlap(cluster_1, cluster_2):
    '''
    Calculate maximal percentage overlap of two N dimensional squares approximated by line intervals in each dimension.
    Parameters
    ----------
    cluster_1: torch.Tensor
        Tensor of shape 2 x N describing the start and end points for each dimension interval of the first square.
    cluster_2: torch.Tensor
        Tensor of shape 2 x N describing the start and end points for each dimension interval of the second square.
    Returns
    -------
    Bigger overlap percentage of the two
    '''

    overlap = torch.min(cluster_1[1], cluster_2[1]) - torch.max(cluster_1[0], cluster_2[0])
    overlap = torch.max(torch.zeros_like(overlap), overlap)
    frac_1 = torch.mean(overlap / (cluster_1[1] - cluster_1[0]))
    frac_2 = torch.mean(overlap / (cluster_2[1] - cluster_2[0]))
    return torch.max(frac_1, frac_2)


def get_certainty(latent, ct_mean, ct_var):
    '''
    Calculate maximal percentage overlap of two N dimensional squares approximated by line intervals in each dimension.
    Parameters
    ----------
    latent: torch.Tensor
        Tensor of shape M x N describing the latent space embedding of M different cells.
    ct_cluster: torch.Tensor
        Tensor of shape 2 x N describing the start and end points for each dimension interval of the cell type cluster.
    Returns
    -------
    Certainty for every cell belonging to this cluster
    '''
    distance = torch.pow(latent - ct_mean.expand(latent.size(0), latent.size(1)), 2)
    overlap = distance / (2 * ct_var.expand(latent.size(0), latent.size(1)))
    overlap = overlap.mean(1)
    overlap = torch.max(torch.zeros_like(overlap), 1 - overlap)
    return overlap


def get_certainty_hd(latent, ct_mean, ct_var):
    '''
    Calculate maximal percentage overlap of two N dimensional squares approximated by line intervals in each dimension.
    Parameters
    ----------
    latent: torch.Tensor
        Tensor of shape M x N describing the latent space embedding of M different cells.
    ct_cluster: torch.Tensor
        Tensor of shape 2 x N describing the start and end points for each dimension interval of the cell type cluster.
    Returns
    -------
    Certainty for every cell belonging to this cluster
    '''
    n = latent.size(0)
    m = ct_mean.size(0)
    d = latent.size(1)
    if d != ct_mean.size(1):
        raise Exception

    latent = latent.unsqueeze(1).expand(n, m, d)
    ct_mean = ct_mean.unsqueeze(0).expand(n, m, d)
    ct_var = ct_var.unsqueeze(0).expand(n, m, d)
    distance = torch.pow(latent - ct_mean, 2)
    overlap = distance / (2 * ct_var)
    overlap = overlap.mean(2)
    probs, preds = torch.max(1 - overlap, dim=1)
    probs = torch.max(torch.zeros_like(probs), probs)
    return probs, preds

'''
a_landmark = torch.tensor([0,0,0])
a_landmark_var = torch.tensor([2,2,2])
b_landmark = torch.tensor([2,2,2])
b_landmark_var = torch.tensor([1,1,1])
test = a_landmark - a_landmark_var
a = torch.stack((a_landmark - a_landmark_var, a_landmark + a_landmark_var))
b = torch.stack((b_landmark - b_landmark_var, b_landmark + b_landmark_var))
frac = get_overlap(a, b)
print(frac)
'''
latent = torch.tensor([
    [0,0,0],
    [1,1,1],
    [1,-1,-1],
    [2,0,0]],
    dtype=torch.float64)
a_landmark = torch.tensor([[0,0,0],[1,1,1],[2,2,2]], dtype=torch.float64)
a_landmark_var = torch.tensor([[1,1,1],[1,1,1],[1,1,1]], dtype=torch.float64)
print(get_certainty_hd(latent, a_landmark, a_landmark_var))
