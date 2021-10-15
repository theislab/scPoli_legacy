import argparse

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

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from scarches.dataset.trvae.data_handling import remove_sparsity
from sklearn.metrics import classification_report

from lataq.models import TRANVAE

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

RESULTS_DIR = args.results_dir
DATA_DIR = args.data_dir
experiments = [args.experiment]
test_nrs = [args.testnr]

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


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
                f'{DATA_DIR}/benchmark_pbmc_shrinked.h5ad')
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

        tranvae = TRANVAE.load_query_data(
            adata=target_adata,
            reference_model=f'{RESULTS_DIR}/{experiment}/{test_nr}_ref_model',
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
        tranvae.save(f'{RESULTS_DIR}/{experiment}/{test_nr}_model',
                     overwrite=True)
        text_file_t = open(
            f'{RESULTS_DIR}/{experiment}/{test_nr}_query_runtime.txt', "w")
        m = text_file_t.write(str(q_time))
        text_file_t.close()

        # EVAL UNLABELED
        data_latent = tranvae.get_latent()
        adata_latent = sc.AnnData(data_latent)
        adata_latent.obs['batch'] = target_adata.obs[condition_key].tolist()
        results_dict = tranvae.classify(metric=class_metric)
        for i in range(len(cell_type_key)):
            preds = results_dict[i]['preds']
            probs = results_dict[i]['probs']

            text_file_q = open(
                f'{RESULTS_DIR}/{experiment}/{test_nr}_query_acc_report_{i}.txt',
                "w"
                )
            n = text_file_q.write(classification_report(
                y_true=target_adata.obs[cell_type_key[i]],
                y_pred=preds,
                labels=np.array(target_adata.obs[cell_type_key[i]].unique().tolist())
            ))
            text_file_q.close()

            correct_probs = probs[preds == target_adata.obs[cell_type_key[i]]]
            incorrect_probs = probs[preds != target_adata.obs[cell_type_key[i]]]
            data = [correct_probs, incorrect_probs]
            fig, ax = plt.subplots()
            ax.set_title('Default violin plot')
            ax.set_ylabel('Observed values')
            ax.violinplot(data)
            labels = ['Correct', 'Incorrect']
            set_axis_style(ax, labels)
            plt.savefig(
                f'{RESULTS_DIR}/{experiment}/{test_nr}_query_uncertainty_{i}.png',
                bbox_inches='tight')

            checks = np.array(len(target_adata) * ['incorrect'])
            checks[preds == target_adata.obs[cell_type_key[i]]] = 'correct'
            adata_latent.obs[cell_type_key[i]] = target_adata.obs[cell_type_key[i]].tolist()
            adata_latent.obs[f'{cell_type_key[i]}_pred'] = preds.tolist()
            adata_latent.obs[f'{cell_type_key[i]}_bool'] = checks.tolist()

        adata_latent.write_h5ad(filename=(
            f'{RESULTS_DIR}/{experiment}/{test_nr}_query_adata.h5ad'))
        sc.pp.neighbors(adata_latent, n_neighbors=8)
        sc.tl.leiden(adata_latent)
        sc.tl.umap(adata_latent)

        sc.pl.umap(adata_latent,
                   color=['batch'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            f'{RESULTS_DIR}/{experiment}/{test_nr}_query_umap_batch.png',
            bbox_inches='tight'
            )
        plt.close()

        for key in cell_type_key:
            sc.pl.umap(adata_latent,
                       color=[key, f'{key}_pred', f'{key}_bool'],
                       frameon=False,
                       wspace=0.6,
                       show=False
                       )
            plt.savefig(
                f'{RESULTS_DIR}/{experiment}/{test_nr}_query_umap_{key}.png',
                bbox_inches='tight'
                )
            plt.close()

        # EVAL FULL
        data_latent = tranvae.get_latent(adata.X, adata.obs[condition_key])
        adata_latent = sc.AnnData(data_latent)
        adata_latent.obs['batch'] = adata.obs[condition_key].tolist()
        results_dict = tranvae.classify(adata.X, adata.obs[condition_key], metric=class_metric)
        for i in range(len(cell_type_key)):
            preds = results_dict[i]['preds']
            probs = results_dict[i]['probs']
            text_file_f = open(
                f'{RESULTS_DIR}/{experiment}/{test_nr}_full_acc_report_{i}.txt'), "w")
            n = text_file_f.write(classification_report(y_true=adata.obs[cell_type_key[i]], y_pred=preds))
            text_file_f.close()

            correct_probs = probs[preds == adata.obs[cell_type_key[i]]]
            incorrect_probs = probs[preds != adata.obs[cell_type_key[i]]]
            data = [correct_probs, incorrect_probs]
            fig, ax = plt.subplots()
            ax.set_title('Default violin plot')
            ax.set_ylabel('Observed values')
            ax.violinplot(data)
            labels = ['Correct', 'Incorrect']
            set_axis_style(ax, labels)
            plt.savefig(
                f'{RESULTS_DIR}/{experiment}/{test_nr}_full_uncertainty_{i}.png'),
                bbox_inches='tight')

            checks = np.array(len(adata) * ['incorrect'])
            checks[preds == adata.obs[cell_type_key[i]]] = 'correct'
            adata_latent.obs[cell_type_key[i]] = adata.obs[cell_type_key[i]].tolist()
            adata_latent.obs[f'{cell_type_key[i]}_pred'] = preds.tolist()
            adata_latent.obs[f'{cell_type_key[i]}_bool'] = checks.tolist()

        adata_latent.write_h5ad(filename=(
            f'{RESULTS_DIR}/{experiment}/{test_nr}_full_adata.h5ad'))
        sc.pp.neighbors(adata_latent, n_neighbors=8)
        sc.tl.leiden(adata_latent)
        sc.tl.umap(adata_latent)

        sc.pl.umap(adata_latent,
                   color=['batch'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            f'{RESULTS_DIR}/{experiment}/{test_nr}_full_umap_batch.png',
            bbox_inches='tight'
            )
        plt.close()

        for key in cell_type_key:
            sc.pl.umap(adata_latent,
                       color=[key, f'{key}_pred', f'{key}_bool'],
                       frameon=False,
                       wspace=0.6,
                       show=False
                       )
            plt.savefig(
                f'{RESULTS_DIR}/{experiment}/{test_nr}_full_umap_{key}.png',
                bbox_inches='tight'
            )
            plt.close()