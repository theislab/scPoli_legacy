import scanpy as sc
import torch
import os
from tranvae.model import TRANVAE
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

test_nr = 3
experiment = "pancreas"


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
source_adata = adata[adata.obs.study.isin(reference)]
target_adata = adata[adata.obs.study.isin(query)]

tranvae = TRANVAE.load(
    dir_path=os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_surg/surg_model'),
    adata=adata
)

preds, probs = tranvae.classify(target_adata.X, target_adata.obs[condition_key], metric="var")
print('Distance Classifier:', np.mean(preds == target_adata.obs[cell_type_key]))
text_file = open(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_surg/acc_report.txt'), "w")
n = text_file.write(classification_report(y_true=target_adata.obs[cell_type_key], y_pred=preds))
text_file.close()

correct_probs = probs[preds == target_adata.obs[cell_type_key]]
incorrect_probs = probs[preds != target_adata.obs[cell_type_key]]
data = [correct_probs, incorrect_probs]
fig, ax = plt.subplots()
ax.set_title('Default violin plot')
ax.set_ylabel('Observed values')
ax.violinplot(data)
labels = ['Correct', 'Incorrect']
set_axis_style(ax, labels)
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_surg/uncertainty.png'), bbox_inches='tight')

x,y,c,p = tranvae.get_landmarks_info()
y_l = np.unique(y).tolist()
c_l = np.unique(c).tolist()
y_uniq = adata.obs[cell_type_key].unique().tolist()
y_uniq_m = tranvae.cell_types_

preds, probs = tranvae.classify(metric="var")

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
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_surg/umap_full_tranvae_batch_l.png'),
            bbox_inches='tight')

sc.pl.umap(adata_latent,
           color=['celltype'],
           groups=y_l,
           frameon=False,
           wspace=0.6,
           size=50,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_surg/umap_full_tranvae_ct_l.png'),
            bbox_inches='tight')

sc.pl.umap(adata_latent,
           color=['celltype'],
           groups=y_uniq,
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_surg/umap_full_tranvae_ct.png'),
            bbox_inches='tight')

sc.pl.umap(adata_latent,
           color=['predictions'],
           groups=y_uniq_m,
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_testing/{experiment}_surg/umap_full_tranvae_pred.png'),
            bbox_inches='tight')
