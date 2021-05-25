import numpy as np
import scanpy as sc
import torch
import os
from scarches.dataset.trvae.data_handling import remove_sparsity
from tranvae.model import TRANVAE

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

experiment = "pancreas"
test_nr = 3
cells_per_ct = 500
skip_celltype = "Pancreas Delta"
figure = 1
model_name = "model"
class_metric = "dist"

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

if figure == 1:
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

    tranvae = TRANVAE.load(os.path.expanduser(f'~/Documents/tranvae_testing/figure_{figure}/{model_name}'), source_adata)

    x,y,c,p = tranvae.get_landmarks_info(metric=class_metric, threshold=0.5)
    print(p)
    print(y)
    y_l = np.unique(y).tolist()
    c_l = np.unique(c).tolist()
    y_uniq = source_adata.obs[cell_type_key].unique().tolist()
    y_uniq_m = tranvae.cell_types_

    preds, probs = tranvae.classify(metric=class_metric)
    data_latent = tranvae.get_latent()
    data_extended = np.concatenate((data_latent, x))
    adata_latent = sc.AnnData(data_extended)
    adata_latent.obs['celltype'] = source_adata.obs[cell_type_key].tolist() + y.tolist()
    adata_latent.obs['batch'] = source_adata.obs[condition_key].tolist() + c.tolist()
    adata_latent.obs['predictions'] = preds.tolist() + y.tolist()
    adata_latent.write_h5ad(filename=os.path.expanduser(f'~/Documents/tranvae_testing/figure_{figure}/ref_data.h5ad'))