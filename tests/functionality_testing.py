import torch
import numpy as np

classes = torch.tensor([3,4,5])
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