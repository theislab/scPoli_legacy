import scanpy as sc
import torch
import os
import numpy as np
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
from tranvae.model import TRANVAE

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

test_nr = 3
condition_key = "study"
cell_type_key = "cell_type"


surgery_epochs = 500

early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "mode": "min",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/pancreas_normalized.h5ad'))
adata = adata_all.raw.to_adata()
adata = remove_sparsity(adata)

if test_nr == 1:
    reference = ['Pancreas inDrop']
    query = ['Pancreas SS2', 'Pancreas CelSeq2', 'Pancreas CelSeq', 'Pancreas Fluidigm C1']
elif test_nr == 2:
    reference = ['Pancreas inDrop', 'Pancreas SS2']
    query = ['Pancreas CelSeq2', 'Pancreas CelSeq', 'Pancreas Fluidigm C1']
elif test_nr == 3:
    reference = ['Pancreas inDrop', 'Pancreas SS2', 'Pancreas CelSeq2']
    query = ['Pancreas CelSeq', 'Pancreas Fluidigm C1']
elif test_nr == 4:
    reference = ['Pancreas inDrop', 'Pancreas SS2', 'Pancreas CelSeq2', 'Pancreas CelSeq']
    query = ['Pancreas Fluidigm C1']
elif test_nr == 5:
    reference = ['Pancreas inDrop', 'Pancreas SS2', 'Pancreas CelSeq2', 'Pancreas CelSeq', 'Pancreas Fluidigm C1']
    query = []

source_adata = adata[adata.obs.study.isin(reference)]
target_adata = adata[adata.obs.study.isin(query)]

ref_path = os.path.expanduser(f'~/Documents/tranvae_testing/pancreas_surg/reference_model')

new_tranvae = TRANVAE.load_query_data(
    adata=target_adata,
    reference_model=ref_path,
    labeled_indices=[],
    clustering='louvain',
    use_unlabeled_loss=True,
)
new_tranvae.train(
    n_epochs=surgery_epochs,
    early_stopping_kwargs=early_stopping_kwargs,
    eta_epoch_anneal=100,
    eta=1000,
    weight_decay=0,
)

query_latent = sc.AnnData(new_tranvae.get_latent())
query_latent.obs['celltype'] = target_adata.obs[cell_type_key].tolist()
query_latent.obs['batch'] = target_adata.obs[condition_key].tolist()
pred_names, probs = new_tranvae.check_for_unseen()
print(pred_names)
print(probs)
preds_dist, _ = new_tranvae.classify()
print('Distance Classifier:', np.mean(preds_dist == target_adata.obs[cell_type_key]))

sc.pp.neighbors(query_latent)
sc.tl.leiden(query_latent)
sc.tl.umap(query_latent)
plt.figure()
sc.pl.umap(query_latent,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/pancreas_surg/umap_surg_tranvae.png'), bbox_inches='tight')

full_latent = sc.AnnData(new_tranvae.get_latent(adata.X, adata.obs[condition_key]))
full_latent.obs['celltype'] = adata.obs[cell_type_key].tolist()
full_latent.obs['batch'] = adata.obs[condition_key].tolist()

sc.pp.neighbors(full_latent)
sc.tl.leiden(full_latent)
sc.tl.umap(full_latent)

plt.figure()
sc.pl.umap(
    full_latent,
    color=["batch", "celltype"],
    frameon=False,
    ncols=1,
    show=False
)
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/pancreas_surg/umap_full_tranvae.png'), bbox_inches='tight')

surg_path = os.path.expanduser(f'~/Documents/tranvae_testing/pancreas_surg/surg_model')
new_tranvae.save(surg_path, overwrite=True)
