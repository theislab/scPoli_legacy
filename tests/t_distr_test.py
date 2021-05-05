import torch
from torch.distributions import Normal
from tranvae.trainers._utils import t_dist, target_distribution, kl_loss


q_distr = Normal(0, 1)
p_distr = Normal(0, 10000)
data = q_distr.sample((500, 200))
centroid = p_distr.sample((10, 200))

q = t_dist(data, centroid, alpha=1)
p = target_distribution(q)
loss = kl_loss(q,p)
print(loss)
