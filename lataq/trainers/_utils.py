import torch


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


def t_dist(x, y, alpha):
    """ student t-distribution, as same as used in t-SNE algorithm.
             q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
    Arguments:
        inputs: the variable containing data, shape=(n_samples, n_features)
    Return:
        q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    distances = torch.pow(x - y, 2).sum(2) / alpha

    q = 1.0 / (1.0 + distances)
    q = torch.pow(q, (alpha + 1.0) / 2.0)
    q = (q.T / q.sum(1)).T
    return q


def target_distribution(q):
    weight = torch.pow(q, 2) / q.sum(0)
    return (weight.T / weight.sum(1)).T


def kl_loss(p, q):
    return (p * torch.log(p / q)).sum(1).mean()


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    d = x.size(1)
    m = y.size(0)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)