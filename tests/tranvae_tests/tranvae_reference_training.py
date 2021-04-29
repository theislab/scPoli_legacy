import scanpy as sc
import torch
import os
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
from tranvae.model import TRANVAE

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

test_nr = 3
condition_key = "study"
cell_type_key = "cell_type"

tranvae_epochs = 500

early_stopping_kwargs = {
    "early_stopping_metric": "val_accuracy",
    "mode": "max",
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

tranvae = TRANVAE(
    adata=source_adata,
    condition_key=condition_key,
    cell_type_key=cell_type_key,
    hidden_layer_sizes=[128, 128],
    use_mmd=False,
    clustering='louvain',
)
tranvae.model.load_state_dict(torch.load(
    os.path.expanduser(f'~/Documents/tranvae_testing/pancreas_surg/reference_model_state_dict')))

tranvae.train(
    n_epochs=tranvae_epochs,
    early_stopping_kwargs=early_stopping_kwargs,
    eta_epoch_anneal=100,
    eta=1000,
    tau=0,
)
ref_path = os.path.expanduser(f'~/Documents/tranvae_testing/pancreas_surg/reference_model')
tranvae.save(ref_path, overwrite=True)

adata_latent = sc.AnnData(tranvae.get_latent())
adata_latent.obs['celltype'] = source_adata.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = source_adata.obs[condition_key].tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/pancreas_surg/umap_ref_tranvae.png'), bbox_inches='tight')