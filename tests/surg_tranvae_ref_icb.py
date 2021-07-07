iimport argparse

parser = argparse.ArgumentParser(description='TRANVAE testing')
parser.add_argument(
    '--experiment', 
    type=str,
    help='Dataset of test'
)
parser.add_argument(
    '--testnr', 
    type=int,
    help='Number of test',
    default=10
)

parser.add_argument(
    '--data_dir', 
    type=str,
    help='Number of test',
    default='../data'
)

parser.add_argument(
    '--results_dir', 
    type=str,
    help='Number of test',
    default='../tranvae_benchmarks/batchwise/surg'
)

args = parser.parse_args()
print(args)

import os
import time
import numpy as np
import scanpy as sc
import torch

import matplotlib.pyplot as plt
from scarches.dataset.trvae.data_handling import remove_sparsity
from tranvae.model import TRANVAE
from sklearn.metrics import classification_report

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

RESULTS_DIR = args.results_dir
DATA_DIR = args.data_dir
experiments = [args.experiment]
test_nrs = [args.testnr]

# Model Params
latent_dim = 10
use_mmd = False

# Training Params
tranvae_epochs = 500
pretraining_epochs = 0
alpha_epoch_anneal = 1e6
eta = 1
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
cell_type_key = ["cell_type"]
for experiment in experiments:
    for test_nr in test_nrs:
        if experiment == "pancreas":
            adata = sc.read(f'{DATA_DIR}/benchmark_pancreas_shrinked.h5ad')
            condition_key = "study"
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
                reference = ['Pancreas inDrop', 'Pancreas SS2', 'Pancreas CelSeq2', 'Pancreas CelSeq',
                             'Pancreas Fluidigm C1']
                query = []
            elif test_nr == 10:
                reference = ["inDrop1", "inDrop2", "inDrop3", "inDrop4", "fluidigmc1", "smartseq2", "smarter"]
                query = ["celseq", "celseq2"]
        if experiment == "pbmc":
            adata = sc.read(
                f'{DATA_DIR}/benchmark_pbmc_shrinked.h5ad'
                )
            condition_key = 'condition'
            if test_nr == 1:
                reference = ['Oetjen']
                query = ['10X', 'Sun', 'Freytag']
            elif test_nr == 2:
                reference = ['Oetjen', '10X']
                query = ['Sun', 'Freytag']
            elif test_nr == 3:
                reference = ['Oetjen', '10X', 'Sun']
                query = ['Freytag']
            elif test_nr == 4:
                reference = ['Oetjen', '10X', 'Sun', 'Freytag']
                query = []
            elif test_nr == 10:
                reference = ['Oetjen', '10X', 'Sun']
                query = ['Freytag']
        if experiment == "brain":
            adata = sc.read(
                f'{DATA_DIR}/benchmark_brain_shrinked.h5ad')
            condition_key = "study"
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
            elif test_nr == 10:
                reference = ['Rosenberg', 'Saunders']
                query = ['Zeisel', 'Tabula_muris']
        if experiment == "scvelo":
            adata = sc.read(
                f'{DATA_DIR}/benchmark_scvelo_shrinked.h5ad')
            condition_key = "study"
            if test_nr == 1:
                reference = ['12.5']
                query = ['13.5', '14.5', '15.5']
            elif test_nr == 2:
                reference = ['12.5', '13.5']
                query = ['14.5', '15.5']
            elif test_nr == 3:
                reference = ['12.5', '13.5', '14.5']
                query = ['15.5']
            elif test_nr == 4:
                reference = ['12.5', '13.5', '14.5', '15.5']
                query = []
            elif test_nr == 10:
                reference = ['12.5', '13.5']
                query = ['14.5', '15.5']
        if experiment == "lung":
            adata = sc.read(
                f'{DATA_DIR}/benchmark_lung_shrinked.h5ad')
            condition_key = "condition"
            if test_nr == 1:
                reference = ['Dropseq_transplant', '10x_Biopsy']
                query = ['10x_Transplant']
            elif test_nr == 10:
                reference = ['Dropseq_transplant', '10x_Biopsy']
                query = ['10x_Transplant']
        if experiment == "tumor":
            adata = sc.read(
                f'{DATA_DIR}/benchmark_tumor_shrinked.h5ad')
            condition_key = "study"
            if test_nr == 10:
                reference = ['breast', 'colorectal', 'liver2', 'liver1', 'lung1', 'lung2', 'multiple', 'ovary',
                             'pancreas', 'skin']
                query = ['melanoma1', 'melanoma2', 'uveal melanoma']
        if experiment == "lung_h":
            adata = sc.read(
                f'{DATA_DIR}/adata_lung_subsampled.h5ad')
            condition_key = "study"
            cell_type_key = ["ann_level_1", "ann_level_2"]
            if test_nr == 10:
                reference = ["Stanford_Krasnow_bioRxivTravaglini", "Misharin_new"]
                query = ["Vanderbilt_Kropski_bioRxivHabermann_vand", "Sanger_Teichmann_2019VieiraBraga"]

        adata = remove_sparsity(adata)
        source_adata = adata[adata.obs.study.isin(reference)].copy()
        target_adata = adata[adata.obs.study.isin(query)].copy()

        tranvae = TRANVAE(
            adata=source_adata,
            condition_key=condition_key,
            cell_type_keys=cell_type_key,
            hidden_layer_sizes=[128, 128],
            latent_dim=latent_dim,
            use_mmd=use_mmd,
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
        ref_path = f'{RESULTS_DIR}/{experiment}/{test_nr}_ref_model'
        tranvae.save(ref_path, overwrite=True)

        text_file_t = open(
            f'{RESULTS_DIR}/{experiment}/{test_nr}_ref_runtime.txt', "w"
            )
        m = text_file_t.write(str(ref_time))
        text_file_t.close()