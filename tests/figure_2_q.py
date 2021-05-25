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
experiment = "brain"
test_nr = 3

# Training Params
tranvae_epochs = 500
pretraining_epochs = 100
eta = 10
tau = 0
clustering_res = 2
labeled_loss_metric = "dist"
unlabeled_loss_metric = "dist"
class_metric = "dist"


early_stopping_kwargs = {
    "early_stopping_metric": "val_classifier_loss",
    "mode": "min",
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
source_adata = adata[adata.obs.study.isin(reference)].copy()
target_adata = adata[adata.obs.study.isin(query)].copy()

tranvae = TRANVAE.load_query_data(
    adata=target_adata,
    reference_model=os.path.expanduser(f'~/Documents/tranvae_testing/figure_2/{experiment}/{test_nr}_ref_model'),
    labeled_indices=[],
)
q_time = time.time()
tranvae.train(
    n_epochs=tranvae_epochs,
    early_stopping_kwargs=early_stopping_kwargs,
    pretraining_epochs=pretraining_epochs,
    eta=eta,
    tau=tau,
    weight_decay=0,
    clustering_res=clustering_res,
    labeled_loss_metric=labeled_loss_metric,
    unlabeled_loss_metric=unlabeled_loss_metric
)
q_time = time.time() - q_time
tranvae.save(os.path.expanduser(f'~/Documents/tranvae_testing/figure_2/{experiment}/{test_nr}_query_model'),
             overwrite=True)

adata_latent = sc.AnnData(tranvae.get_latent())
adata_latent.obs['celltype'] = target_adata.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = target_adata.obs[condition_key].tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/figure_2/{experiment}/{test_nr}_query_umap_tranvae.png'),
            bbox_inches='tight')

preds, probs = tranvae.classify(metric=class_metric)
print('Distance Classifier:', np.mean(preds == target_adata.obs[cell_type_key]))
text_file = open(os.path.expanduser(f'~/Documents/tranvae_testing/figure_2/{experiment}/{test_nr}_query_acc_report.txt'), "w")
n = text_file.write(classification_report(y_true=target_adata.obs[cell_type_key], y_pred=preds))
text_file.close()
text_file_t = open(os.path.expanduser(f'~/Documents/tranvae_testing/figure_2/{experiment}/{test_nr}_query_runtime.txt'), "w")
m = text_file_t.write(str(q_time))
text_file_t.close()

correct_probs = probs[preds == target_adata.obs[cell_type_key]]
incorrect_probs = probs[preds != target_adata.obs[cell_type_key]]
data = [correct_probs, incorrect_probs]
fig, ax = plt.subplots()
ax.set_title('Default violin plot')
ax.set_ylabel('Observed values')
ax.violinplot(data)
labels = ['Correct', 'Incorrect']
set_axis_style(ax, labels)
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/figure_2/{experiment}/{test_nr}_query_uncertainty.png'),
            bbox_inches='tight')

adata_latent = sc.AnnData(tranvae.get_latent(adata.X, adata.obs[condition_key],))
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
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/figure_2/{experiment}/{test_nr}_query_umap_tranvae_full.png'),
            bbox_inches='tight')