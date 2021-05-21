import numpy as np
import scanpy as sc
import torch
import os
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
from tranvae.model import TRANVAE
import time
from sklearn.metrics import classification_report

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


# Experiment Params
experiment = "pancreas"
unlabeled_strat = "ct"
test_nr = 3
cells_per_ct = 500

# Model Params
latent_dim = 20
use_mmd = False

# Training Params
tranvae_epochs = 500
pretraining_epochs = 200
alpha_epoch_anneal = 100
eta = 100
tau = 1
clustering_res = 2
labeled_loss_metric = "seurat"
unlabeled_loss_metric = "seurat"
class_metric = "seurat"


early_stopping_kwargs = {
    "early_stopping_metric": "val_accuracy",
    "mode": "max",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}


if experiment == "pancreas":
    adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/pancreas_normalized.h5ad'))
    condition_key = "study"
    cell_type_key = "cell_type"
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
if experiment == "pbmc":
    adata_all = sc.read(os.path.expanduser(
        f'~/Documents/benchmarking_datasets/Immune_ALL_human_wo_villani_rqr_normalized_hvg.h5ad'))
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
if experiment == "brain":
    adata_all = sc.read(
        os.path.expanduser(f'~/Documents/benchmarking_datasets/mouse_brain_subsampled_normalized_hvg.h5ad'))
    condition_key = "study"
    cell_type_key = "cell_type"
    if test_nr == 1:
        reference = ['Rosenberg']
        query = ['Saunders', 'Zeisel', 'Tabula_muris']
    elif test_nr == 2:
        reference = ['Rosenberg', 'Saunders']
        query = ['Zeisel', 'Tabula_muris']
    elif test_nr == 3:
        reference = ['Rosenberg', 'Saunders', 'Zeisel']
        query = ['Tabula_muris']
    elif test_nr == 4:
        reference = ['Rosenberg', 'Saunders', 'Zeisel', 'Tabula_muris']
        query = []

adata = adata_all.raw.to_adata()
adata = remove_sparsity(adata)

indices = np.arange(len(adata))
if unlabeled_strat == "batch":
    labeled_ind = indices[adata.obs.study.isin(reference)].tolist()
    labeled_adata = adata[adata.obs.study.isin(reference)].copy()
    unlabeled_adata = adata[adata.obs.study.isin(query)].copy()
if unlabeled_strat == "ct":
    labeled_ind = []
    cts = adata.obs[cell_type_key].unique().tolist()
    for celltype in cts:
        ct_indices = indices[adata.obs[cell_type_key].isin([celltype])]
        ct_sel_ind = np.random.choice(ct_indices, size=cells_per_ct, replace=False)
        labeled_ind += ct_sel_ind.tolist()
        print(celltype, len(ct_indices), len(ct_sel_ind), len(labeled_ind))
    unlabeled_ind = np.delete(indices, labeled_ind).tolist()
    labeled_adata = adata[labeled_ind].copy()
    unlabeled_adata = adata[unlabeled_ind].copy()

tranvae = TRANVAE(
    adata=adata,
    condition_key=condition_key,
    cell_type_key=cell_type_key,
    hidden_layer_sizes=[128, 128],
    latent_dim=latent_dim,
    use_mmd=use_mmd,
    labeled_indices=labeled_ind
)
ref_time = time.time()
tranvae.train(
    n_epochs=tranvae_epochs,
    early_stopping_kwargs=early_stopping_kwargs,
    pretraining_epochs=pretraining_epochs,
    alpha_epoch_anneal=alpha_epoch_anneal,
    eta=eta,
    tau=tau,
    clustering_res=clustering_res,
    labeled_loss_metric=labeled_loss_metric,
    unlabeled_loss_metric=unlabeled_loss_metric
)
ref_time = time.time() - ref_time
ref_path = os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_semi/model')
tranvae.save(ref_path, overwrite=True)

adata_latent = sc.AnnData(tranvae.get_latent())
adata_latent.obs['celltype'] = adata.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = adata.obs[condition_key].tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_semi/umap_tranvae.png'), bbox_inches='tight')

preds, probs = tranvae.classify(unlabeled_adata.X, unlabeled_adata.obs[condition_key], metric=class_metric)
print('Distance Classifier:', np.mean(preds == unlabeled_adata.obs[cell_type_key]))
text_file = open(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_semi/acc_report.txt'), "w")
n = text_file.write(classification_report(y_true=unlabeled_adata.obs[cell_type_key], y_pred=preds))
text_file.close()
text_file_t = open(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_semi/runtime.txt'), "w")
m = text_file_t.write(str(ref_time))
text_file_t.close()

correct_probs = probs[preds == unlabeled_adata.obs[cell_type_key]]
incorrect_probs = probs[preds != unlabeled_adata.obs[cell_type_key]]
data = [correct_probs, incorrect_probs]
fig, ax = plt.subplots()
ax.set_title('Default violin plot')
ax.set_ylabel('Observed values')
ax.violinplot(data)
labels = ['Correct', 'Incorrect']
set_axis_style(ax, labels)
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_semi/uncertainty.png'), bbox_inches='tight')

x,y,c,p = tranvae.get_landmarks_info(metric=class_metric)
print(p)
print(y)
y_l = np.unique(y).tolist()
c_l = np.unique(c).tolist()
y_uniq = adata.obs[cell_type_key].unique().tolist()
y_uniq_m = tranvae.cell_types_

preds, probs = tranvae.classify(metric=class_metric)
data_latent = tranvae.get_latent()
data_extended = np.concatenate((data_latent, x))
adata_latent = sc.AnnData(data_extended)
adata_latent.obs['celltype'] = adata.obs[cell_type_key].tolist() + y.tolist()
adata_latent.obs['batch'] = adata.obs[condition_key].tolist() + c.tolist()
adata_latent.obs['predictions'] = preds.tolist() + y.tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch'],
           groups=c_l,
           frameon=False,
           wspace=0.6,
           size=50,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_semi/umap_full_tranvae_batch_l.png'),
            bbox_inches='tight')

sc.pl.umap(adata_latent,
           color=['celltype'],
           groups=y_l,
           frameon=False,
           wspace=0.6,
           size=50,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_semi/umap_full_tranvae_ct_l.png'),
            bbox_inches='tight')

sc.pl.umap(adata_latent,
           color=['celltype'],
           groups=y_uniq,
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_semi/umap_full_tranvae_ct.png'),
            bbox_inches='tight')

sc.pl.umap(adata_latent,
           color=['predictions'],
           groups=y_uniq_m,
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_semi/umap_full_tranvae_pred.png'),
            bbox_inches='tight')