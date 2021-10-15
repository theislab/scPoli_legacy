import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal

# Check 0 determinant behavior#
size = 20
t = torch.tensor(
    [
        [0.003, 0.003, 0.002, -0.000, 0.002, -0.003, 0.003, -0.003, -0.001, 0.001],
        [0.003, 0.005, 0.002, 0.000, 0.003, -0.003, 0.004, -0.004, -0.002, 0.000],
        [0.002, 0.002, 0.001, 0.000, 0.001, -0.001, 0.001, -0.001, -0.001, 0.000],
        [-0.000, 0.000, 0.000, 0.002, -0.001, 0.002, -0.001, -0.000, -0.001, -0.000],
        [0.002, 0.003, 0.001, -0.001, 0.003, -0.003, 0.004, -0.003, -0.001, 0.000],
        [-0.003, -0.003, -0.001, 0.002, -0.003, 0.006, -0.005, 0.003, -0.001, -0.001],
        [0.003, 0.004, 0.001, -0.001, 0.004, -0.005, 0.006, -0.004, -0.001, 0.001],
        [-0.003, -0.004, -0.001, -0.000, -0.003, 0.003, -0.004, 0.004, 0.002, -0.000],
        [-0.001, -0.002, -0.001, -0.001, -0.001, -0.001, -0.001, 0.002, 0.003, 0.001],
        [0.001, 0.000, 0.000, -0.000, 0.000, -0.001, 0.001, -0.000, 0.001, 0.001],
    ]
)

t = t.type(torch.DoubleTensor)
t_eps = t + torch.eye(10) * 1e-3

c = torch.tensor(
    [
        [0.020, -0.002, 0.003, -0.000, -0.011, 0.007, -0.004, 0.007, -0.002, -0.013],
        [-0.002, 0.023, -0.004, -0.001, 0.006, -0.012, -0.001, -0.009, -0.004, 0.021],
        [0.003, -0.004, 0.025, -0.001, -0.002, 0.000, 0.000, -0.000, 0.000, -0.006],
        [-0.000, -0.001, -0.001, 0.013, 0.004, 0.006, 0.001, -0.003, -0.002, 0.004],
        [-0.011, 0.006, -0.002, 0.004, 0.028, -0.006, 0.001, -0.001, 0.004, -0.000],
        [0.007, -0.012, 0.000, 0.006, -0.006, 0.017, -0.003, 0.006, 0.003, -0.013],
        [-0.004, -0.001, 0.000, 0.001, 0.001, -0.003, 0.009, -0.000, 0.003, -0.001],
        [0.007, -0.009, -0.000, -0.003, -0.001, 0.006, -0.000, 0.049, -0.002, -0.013],
        [-0.002, -0.004, 0.000, -0.002, 0.004, 0.003, 0.003, -0.002, 0.008, -0.011],
        [-0.013, 0.021, -0.006, 0.004, -0.000, -0.013, -0.001, -0.013, -0.011, 0.046],
    ]
)
c = torch.eye(size)
c = c.type(torch.DoubleTensor)
c_eps = c + torch.eye(size) * 1e-16
mean = torch.zeros(size)

m = MultivariateNormal(mean, t_eps, validate_args=False)
# n = LowRankMultivariateNormal(mean, t, cov_diag=torch.diag(t),validate_args=False)

o = MultivariateNormal(mean, c, validate_args=False)
p = LowRankMultivariateNormal(mean, c, cov_diag=torch.diag(c), validate_args=False)
q = MultivariateNormal(mean, c_eps, validate_args=False)

x = np.random.rand(63, size)
x_t = torch.DoubleTensor(x)

print(o.log_prob(x_t) - p.log_prob(x_t))
print(o.log_prob(x_t) - q.log_prob(x_t))
exit()

# Test covariance behavior
def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


x = np.random.rand(63, 10)
y = np.random.rand(63, 10)
z = np.random.rand(63, 10)
cov_n = np.cov(x.T)
# print(cov_n.shape)
# print(cov_n)

x_t = torch.tensor(x)
y_t = torch.tensor(y)
z_t = torch.tensor(z)
cov_xt = cov(x_t).unsqueeze(0)
cov_yt = cov(y_t).unsqueeze(0)
cov_zt = cov(z_t).unsqueeze(0)
print(cov_xt.size())
cov_xy = torch.cat([cov_xt, cov_yt])
print(cov_xy.size())
cov_xyz = torch.cat([cov_xy, cov_zt])
print(cov_xyz.size())
test_cov = cov_xyz[0].unsqueeze(0)
print(test_cov.size())
cov_xyzx = torch.cat([cov_xyz, test_cov])
print(cov_xyzx.size())
zeros = torch.zeros(1, 10, 10)
cov_xyzxz = torch.cat([cov_xyzx, zeros])
print(cov_xyzxz.size())
mean_x = x_t.mean(0)
mean_y = y_t.mean(0)
mean_z = z_t.mean(0)
probs = []
x_distr = MultivariateNormal(mean_x, cov_xyzxz[0])
probs.append(x_distr.log_prob(x_t).exp())
y_distr = MultivariateNormal(mean_x, cov_xyzxz[1])
probs.append(y_distr.log_prob(y_t).exp())
z_distr = MultivariateNormal(mean_x, cov_xyzxz[2])
probs.append(z_distr.log_prob(z_t).exp())
probs = torch.stack(probs)
print(probs.sum(0))
probs = (probs / probs.sum(0)).T
probs, preds = torch.max(probs, dim=1)
print(probs.size())
print(preds)
print(cov_xyzxz[0])
exit()

# Test Norm behavior
x = np.random.rand(63, 10)
norm = np.sum(x ** 2, axis=1)
norm = np.resize(norm, (x.shape[1], x.shape[0])).T
x = x / np.sqrt(norm)
print(np.sum(x ** 2, axis=1))
exit()

# Test hyperbolic loss behavior
n_celltypes = 5
n_cells = 63
latent = torch.rand((n_cells, 10))
landmarks = torch.rand((n_celltypes, 10))
labels = torch.randint(0, 1, (n_cells, 1))
uniq_labels = torch.unique(labels, sorted=True).tolist()
if uniq_labels == [-1]:
    print("EXCEPTION")
    exit()

# Transform Landmarks to hyperbolic ideal points
h_landmarks = F.normalize(landmarks, p=2, dim=1)

# Transform latent to hyperbolic space and filter out cells with label == -1 which correspond to "unknown"
transformation_m = (
    (torch.tanh(torch.norm(latent, p=2, dim=1) / 2) / torch.norm(latent, p=2, dim=1))
    .unsqueeze(dim=1)
    .expand(-1, latent.size(1))
)
h_latent = transformation_m * latent
h_latent = h_latent[labels.squeeze(1) != -1, :]

# Get tensor of corresponding landmarks and filter out cells with label == -1 which correspond to "unknown"
corr_land = h_landmarks[labels.squeeze(1), :]
corr_land = corr_land[labels.squeeze(1) != -1, :]

# Buseman loss
b_loss = torch.log(
    torch.norm(corr_land - h_latent, p=2, dim=1) ** 2
    / (1 - torch.norm(h_latent, p=2, dim=1) ** 2)
)

# Overconfidence penalty loss
overconf_loss = torch.log(1 - torch.norm(h_latent, p=2, dim=1) ** 2)

# Calculate overall loss by taking mean of each cell
# TODO:
#  - CHECK IF (h_latent.size(1) + 1) IS ENOUGH AS SCALE OR MAKE NEW PARAM

loss_val = (b_loss - (h_latent.size(1) + 1) * overconf_loss).mean()

# Get classification matrix n_cells x n_cell_types and get the predictions by max
class_m = torch.matmul(
    h_latent
    / torch.norm(h_latent, p=2, dim=1).unsqueeze(dim=1).expand(-1, latent.size(1)),
    h_landmarks.T,
)
class_m = F.normalize(class_m, p=1, dim=1)
probs, preds = torch.max(class_m, dim=1)
print(loss_val)
print(preds)
exit()


a = torch.tensor(
    [[0.9041, 0.0196], [-2.1763, -0.4713], [-2.1763, -0.4713], [-2.1763, -0.4713]]
)
idx = torch.tensor([0, 0, 1, 1, 2, 2, -1, -1, -1])
print(idx != -1)
b = a[idx, :]
c = b[idx != -1, :]

print(b)
print("\n", c)
d = torch.sqrt(torch.sum(a ** 2, dim=1))
print(d)
print(torch.norm(a, p=2, dim=1))
exit()

classes = torch.tensor(
    [[3.0, 4.0, 5.0], [3.0, 5.0, 6.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0]]
)
classes = F.normalize(classes, p=2, dim=1)
classes = classes ** 2
print(classes)
exit()

preds = torch.tensor([0, 2, 1, 2, 0, 1])
preds = classes[preds]
print(preds)
exit()

dictio = dict()
dictio["b"] = [1, 2, 3, 4, 5]
dictio["a"] = [3, 4, 5, 6, 7]
cell_type_keys = ["a", "b"]
cell_types_ = dict()
for cell_type_key in cell_type_keys:
    uniq_cts = dictio[cell_type_key]
    for ct in uniq_cts:
        if ct in cell_types_:
            cell_types_[ct].append(cell_type_key)
        else:
            cell_types_[ct] = [cell_type_key]
print(cell_types_)

landmarks_idx = list()
for i, key in enumerate(cell_types_.keys()):
    if cell_type_keys[1] in cell_types_[key]:
        landmarks_idx.append(i)

print(landmarks_idx)
exit()


unknown_ct_names_ = [3, 4, 5]
for unknown_ct in unknown_ct_names_:
    if unknown_ct in cell_types_:
        del cell_types_[unknown_ct]
print(cell_types_)
model_cell_types = list(cell_types_.keys())
print(model_cell_types)
exit()

model_cts = None
for key in dictio:
    model_cts = model_cts + dictio[key] if model_cts is not None else dictio[key]

print(list(set(model_cts)))
print(dictio)
exit()

X = np.random.normal(size=(8, 10))
X = torch.tensor(X)
labels_1 = np.array((1, 1, 2, 2, 3, 3, 4, 4))
labels_2 = np.array((5, 5, 5, 6, 6, 6, 7, 7))
labels_3 = np.array((8, 8, 8, 8, -1, -1, -1, -1))
labels = np.stack([labels_1, labels_2, labels_3]).T
# labels = np.stack([labels_1]).T
labels = torch.tensor(labels, dtype=torch.long)
unique_labels = labels.unique()
unique_labels = unique_labels[unique_labels != -1]

for value in unique_labels:
    indices = labels.eq(value).nonzero(as_tuple=False)[:, 0]
    landmark = X[indices, :].mean(0).unsqueeze(0)
    print(value, len(indices), landmark)
