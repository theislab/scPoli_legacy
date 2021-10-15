import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from scarches.dataset.trvae.data_handling import remove_sparsity
from sklearn.metrics import classification_report
from tranvae.model import TRANVAE

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction="out")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel("Sample name")


# Experiment Params
experiment = "pancreas"
test_nr = 3
cells_per_ct = 500
skip_celltype = "Pancreas Delta"

class_metric = "dist"

if experiment == "pancreas":
    adata_all = sc.read(
        os.path.expanduser(
            f"~/Documents/benchmarking_datasets/pancreas_normalized.h5ad"
        )
    )
    condition_key = "study"
    cell_type_key = "cell_type"
    if test_nr == 1:
        reference = ["Pancreas inDrop"]
        query = [
            "Pancreas SS2",
            "Pancreas CelSeq2",
            "Pancreas CelSeq",
            "Pancreas Fluidigm C1",
        ]
    elif test_nr == 2:
        reference = ["Pancreas inDrop", "Pancreas SS2"]
        query = ["Pancreas CelSeq2", "Pancreas CelSeq", "Pancreas Fluidigm C1"]
    elif test_nr == 3:
        reference = ["Pancreas inDrop", "Pancreas SS2", "Pancreas CelSeq2"]
        query = ["Pancreas CelSeq", "Pancreas Fluidigm C1"]
    elif test_nr == 4:
        reference = [
            "Pancreas inDrop",
            "Pancreas SS2",
            "Pancreas CelSeq2",
            "Pancreas CelSeq",
        ]
        query = ["Pancreas Fluidigm C1"]
    elif test_nr == 5:
        reference = [
            "Pancreas inDrop",
            "Pancreas SS2",
            "Pancreas CelSeq2",
            "Pancreas CelSeq",
            "Pancreas Fluidigm C1",
        ]
        query = []

adata = adata_all.raw.to_adata()
adata = remove_sparsity(adata)
source_adata = adata[adata.obs.study.isin(reference)].copy()
target_adata = adata[adata.obs.study.isin(query)].copy()

indices = np.arange(len(source_adata))
labeled_ind = []
cts = source_adata.obs[cell_type_key].unique().tolist()
for celltype in cts:
    if celltype == skip_celltype:
        continue
    ct_indices = indices[source_adata.obs[cell_type_key].isin([celltype])]
    ct_sel_ind = np.random.choice(ct_indices, size=cells_per_ct, replace=False)
    labeled_ind += ct_sel_ind.tolist()
    print(celltype, len(ct_indices), len(ct_sel_ind), len(labeled_ind))
unlabeled_ind = np.delete(indices, labeled_ind).tolist()
labeled_adata = source_adata[labeled_ind].copy()
unlabeled_adata = source_adata[unlabeled_ind].copy()

tranvae = TRANVAE.load(
    os.path.expanduser(f"~/Documents/tranvae_testing/figure_1/model"), source_adata
)

x, y, c, p = tranvae.get_landmarks_info(metric=class_metric, threshold=0.5)

tranvae.add_new_cell_type("Pancreas Delta", [6])

tranvae.save(
    os.path.expanduser(f"~/Documents/tranvae_testing/figure_1/extended_model"),
    overwrite=True,
)

preds, probs = tranvae.classify(
    unlabeled_adata.X, unlabeled_adata.obs[condition_key], metric=class_metric
)
print("Distance Classifier:", np.mean(preds == unlabeled_adata.obs[cell_type_key]))
text_file = open(
    os.path.expanduser(f"~/Documents/tranvae_testing/figure_1/extended_acc_report.txt"),
    "w",
)
n = text_file.write(
    classification_report(y_true=unlabeled_adata.obs[cell_type_key], y_pred=preds)
)
text_file.close()

x, y, c, p = tranvae.get_landmarks_info(metric=class_metric, threshold=0.4)
y_l = np.unique(y).tolist()
c_l = np.unique(c).tolist()
y_uniq = source_adata.obs[cell_type_key].unique().tolist()
y_uniq_m = tranvae.cell_types_
print(p)
print(y)

preds, probs = tranvae.classify(metric=class_metric)

data_latent = tranvae.get_latent()
data_extended = np.concatenate((data_latent, x))
adata_latent = sc.AnnData(data_extended)
adata_latent.obs["celltype"] = source_adata.obs[cell_type_key].tolist() + y.tolist()
adata_latent.obs["batch"] = source_adata.obs[condition_key].tolist() + c.tolist()
adata_latent.obs["predictions"] = preds.tolist() + y.tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(
    adata_latent,
    color=["batch"],
    groups=c_l,
    frameon=False,
    wspace=0.6,
    size=50,
    show=False,
)
plt.savefig(
    os.path.expanduser(
        f"~/Documents/tranvae_testing/figure_1/extended_umap_full_tranvae_batch_l.png"
    ),
    bbox_inches="tight",
)

sc.pl.umap(
    adata_latent,
    color=["celltype"],
    groups=y_l,
    frameon=False,
    wspace=0.6,
    size=50,
    show=False,
)
plt.savefig(
    os.path.expanduser(
        f"~/Documents/tranvae_testing/figure_1/extended_umap_full_tranvae_ct_l.png"
    ),
    bbox_inches="tight",
)

sc.pl.umap(
    adata_latent,
    color=["celltype"],
    groups=y_uniq,
    frameon=False,
    wspace=0.6,
    show=False,
)
plt.savefig(
    os.path.expanduser(
        f"~/Documents/tranvae_testing/figure_1/extended_umap_full_tranvae_ct.png"
    ),
    bbox_inches="tight",
)

sc.pl.umap(
    adata_latent,
    color=["predictions"],
    groups=y_uniq_m,
    frameon=False,
    wspace=0.6,
    show=False,
)
plt.savefig(
    os.path.expanduser(
        f"~/Documents/tranvae_testing/figure_1/extended_umap_full_tranvae_pred.png"
    ),
    bbox_inches="tight",
)
