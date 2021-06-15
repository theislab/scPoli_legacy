import torch


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


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
    Calculate certainties for each cell in the latent data belonging to the given cell type clusters.
    Parameters
    ----------
    latent: torch.Tensor
        Tensor of shape M x N describing the N-dim. latent space embedding of M different cells.
    ct_mean: torch.Tensor
        Tensor of shape K x N describing the mean in the N-dim. latent space of K cell type clusters.
    ct_var: torch.Tensor
        Tensor of shape K x N describing the variance in the N-dim. latent space of K cell type clusters.
    Returns
    -------
    Certainty for every cell belonging to every cluster
    '''

    n = latent.size(0)
    m = ct_mean.size(0)
    d = latent.size(1)
    if d != ct_mean.size(1):
        raise Exception

    latent = latent.unsqueeze(1).expand(n, m, d)
    ct_mean = ct_mean.unsqueeze(0).expand(n, m, d)
    ct_var = ct_var.unsqueeze(0).expand(n, m, d)

    # Calculate distance between every cell and landmark
    distance = torch.pow(latent - ct_mean, 2)

    # Calculate percentage how likely it is that the cell belongs to each landmark cluster
    overlap = distance / (15 * ct_var)
    overlap = overlap.mean(2)

    # Choose the predicted cluster by looking at the maximum
    overlap_cor = torch.max(torch.zeros_like(overlap), 1 - overlap)
    overlap_normed = (overlap_cor.T / overlap_cor.sum(1)).T
    probs, preds = torch.max(overlap_normed, dim=1)
    probs[torch.isnan(probs)] = 0

    return probs, preds
