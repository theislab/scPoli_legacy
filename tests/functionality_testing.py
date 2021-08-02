import torch
import numpy as np
import torch.nn.functional as F


n_celltypes = 5
n_cells = 63
latent = torch.rand((n_cells,10))
landmarks = torch.rand((n_celltypes,10))
labels = torch.randint(0, 1, (n_cells,1))
uniq_labels = torch.unique(labels, sorted=True).tolist()
if uniq_labels == [-1]:
    print("EXCEPTION")
    exit()

# Transform Landmarks to hyperbolic ideal points
h_landmarks = F.normalize(landmarks, p=2, dim=1)

# Transform latent to hyperbolic space and filter out cells with label == -1 which correspond to "unknown"
transformation_m = (
        torch.tanh(torch.norm(latent, p=2, dim=1) / 2) / torch.norm(latent, p=2, dim=1)
).unsqueeze(dim=1).expand(-1, latent.size(1))
h_latent = transformation_m * latent
h_latent = h_latent[labels.squeeze(1) != -1, :]

# Get tensor of corresponding landmarks and filter out cells with label == -1 which correspond to "unknown"
corr_land = h_landmarks[labels.squeeze(1), :]
corr_land = corr_land[labels.squeeze(1) != -1, :]

# Buseman loss
b_loss = torch.log(torch.norm(corr_land - h_latent, p=2, dim=1) ** 2 / (1 - torch.norm(h_latent, p=2, dim=1) ** 2))

# Overconfidence penalty loss
overconf_loss = torch.log(1 - torch.norm(h_latent, p=2, dim=1) ** 2)

# Calculate overall loss by taking mean of each cell
# TODO:
#  - CHECK IF (h_latent.size(1) + 1) IS ENOUGH AS SCALE OR MAKE NEW PARAM

loss_val = (b_loss - (h_latent.size(1) + 1) * overconf_loss).mean()

# Get classification matrix n_cells x n_cell_types and get the predictions by max
class_m = torch.matmul(
    h_latent / torch.norm(h_latent, p=2, dim=1).unsqueeze(dim=1).expand(-1, latent.size(1)),
    h_landmarks.T
)
class_m = F.normalize(class_m, p=1, dim=1)
probs, preds = torch.max(class_m, dim=1)
print(loss_val)
print(preds)
exit()


a = torch.tensor([[0.9041, 0.0196],[-2.1763, -0.4713],[-2.1763, -0.4713],[-2.1763, -0.4713]])
idx = torch.tensor([0,0,1,1,2,2,-1,-1,-1])
print(idx != -1)
b = a[idx,:]
c = b[idx != -1,:]

print(b)
print("\n", c)
d = torch.sqrt(torch.sum(a ** 2, dim=1))
print(d)
print(torch.norm(a, p=2, dim=1))
exit()

classes = torch.tensor([[3.,4.,5.],[3.,5.,6.],[5.,6.,7.],[7.,8.,9.]])
classes = F.normalize(classes, p=2, dim=1)
classes = classes ** 2
print(classes)
exit()

preds = torch.tensor([0,2,1,2,0,1])
preds = classes[preds]
print(preds)
exit()

dictio = dict()
dictio['b'] = [1,2,3,4,5]
dictio['a'] = [3,4,5,6,7]
cell_type_keys = ['a', 'b']
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


unknown_ct_names_ = [3,4,5]
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

X = np.random.normal(size=(8,10))
X = torch.tensor(X)
labels_1 = np.array((1,1,2,2,3,3,4,4))
labels_2 = np.array((5,5,5,6,6,6,7,7))
labels_3 = np.array((8,8,8,8,-1,-1,-1,-1))
labels = np.stack([labels_1,labels_2,labels_3]).T
#labels = np.stack([labels_1]).T
labels = torch.tensor(labels, dtype=torch.long)
unique_labels = labels.unique()
unique_labels = unique_labels[unique_labels!=-1]

for value in unique_labels:
    indices = labels.eq(value).nonzero(as_tuple=False)[:,0]
    landmark = X[indices,:].mean(0).unsqueeze(0)
    print(value, len(indices), landmark)