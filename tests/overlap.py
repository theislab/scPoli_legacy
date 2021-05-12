import torch
import numpy as np


test_data = np.random.random((100,5))
test_data = torch.tensor(test_data)
test_mean = test_data.mean(0).unsqueeze(0)
test_var = torch.pow(test_data.std(0), 2)

print(test_mean)
print(test_var)
exit()


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


a_landmark = torch.tensor([0,0,0])
a_landmark_var = torch.tensor([2,2,2])
b_landmark = torch.tensor([2,2,2])
b_landmark_var = torch.tensor([1,1,1])
test = a_landmark - a_landmark_var
a = torch.stack((a_landmark - a_landmark_var, a_landmark + a_landmark_var))
b = torch.stack((b_landmark - b_landmark_var, b_landmark + b_landmark_var))
frac = get_overlap(a, b)
print(frac)
