import scanpy as sc
import torch
import os
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt


sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

test_nr = 3

trvae_epochs = 150

early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 200,
    "reduce_lr": True,
    "lr_patience": 50,
    "lr_factor": 0.1,
}

adata_all = sc.read(os.path.expanduser(
    f'~/Documents/benchmarking_datasets/Immune_ALL_human_wo_villani_rqr_normalized_hvg.h5ad'))
adata = adata_all.copy()
adata = remove_sparsity(adata)
condition_key = 'condition'
cell_type_key = 'final_annotation'
if test_nr == 1:
    reference = ['10X']
    query = ['Oetjen', 'Sun', 'Freytag']
elif test_nr == 2:
    reference = ['10X', 'Oetjen']
    query = ['Sun', 'Freytag']
elif test_nr == 3:
    reference = ['10X', 'Oetjen', 'Sun']
    query = ['Freytag']
elif test_nr == 4:
    reference = ['10X', 'Oetjen', 'Sun', 'Freytag']
    query = []

source_adata = adata[adata.obs.study.isin(reference)]
target_adata = adata[adata.obs.study.isin(query)]

trvae = sca.models.TRVAE(
    adata=source_adata,
    condition_key=condition_key,
    hidden_layer_sizes=[128, 128],
    use_mmd=False
)
trvae.train(
    n_epochs=trvae_epochs,
    alpha_epoch_anneal=100,
    early_stopping_kwargs=early_stopping_kwargs
)
torch.save(
    trvae.model.state_dict(),
    os.path.expanduser(f'~/Documents/tranvae_testing/pbmc_surg/reference_model_state_dict')
)

adata_latent = sc.AnnData(trvae.get_latent())
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
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/pbmc_surg/umap_ref.png'), bbox_inches='tight')
