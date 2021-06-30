import torch
import numpy as np

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