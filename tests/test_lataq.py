import pytest

import scanpy as sc
import pandas as pd
from sklearn.metrics import classification_report
from scarches.dataset.trvae.data_handling import remove_sparsity

from lataq.models import EMBEDCVAE, TRANVAE
from lataq_reproduce.utils import label_encoder
from lataq_reproduce.exp_dict import EXPERIMENT_INFO

@pytest.fixture
def fixed_params():
    return {
        'EPOCHS': 10,
        'N_PRE_EPOCHS': 8,
        'DATA_DIR': '../../lataq_reproduce/data',
        'DATA': 'pancreas.h5ad',
        'EARLY_STOPPING_KWARGS': {
            "early_stopping_metric": "val_landmark_loss",
            "mode": "min",
            "threshold": 0,
            "patience": 20,
            "reduce_lr": True,
            "lr_patience": 13,
            "lr_factor": 0.1,
        },
        'LATENT_DIM': 10,
        'ALPHA_EPOCH_ANNEAL': 1e3,
        'CLUSTERING_RES': 2,
        'HIDDEN_LAYERS': 1,
        'ETA': 1,
    }   


@pytest.mark.parametrize("model", ['embedcvae', 'tranvae'])
@pytest.mark.parametrize("loss_metric", ['dist', 'hyperbolic'])
def test_fatal(fixed_params, model, loss_metric):
    EXP_PARAMS = EXPERIMENT_INFO['pancreas']
    adata = sc.read(f'{fixed_params["DATA_DIR"]}/{fixed_params["DATA"]}')
    condition_key = EXP_PARAMS['condition_key']
    cell_type_key = EXP_PARAMS['cell_type_key']
    reference = EXP_PARAMS['reference']
    query = EXP_PARAMS['query']

    EPOCHS = fixed_params['EPOCHS']
    PRE_EPOCHS = fixed_params['N_PRE_EPOCHS']
    REF_PATH = 'tmp/'
    adata = remove_sparsity(adata)
    source_adata = adata[adata.obs.study.isin(reference)].copy()
    target_adata = adata[adata.obs.study.isin(query)].copy()

    if model == 'embedcvae':
        lataq = EMBEDCVAE(
            adata=source_adata,
            condition_key=condition_key,
            cell_type_keys=cell_type_key,
            hidden_layer_sizes=[32]*int(fixed_params['HIDDEN_LAYERS']),
            latent_dim=int(fixed_params['LATENT_DIM']),
            use_mmd=False,
        )
    elif model == 'tranvae':
        lataq = TRANVAE(
            adata=source_adata,
            condition_key=condition_key,
            cell_type_keys=cell_type_key,
            hidden_layer_sizes=[32]*int(fixed_params['HIDDEN_LAYERS']),
            latent_dim=int(fixed_params['LATENT_DIM']),
            use_mmd=False,
        )
    lataq.train(
        n_epochs=EPOCHS,
        early_stopping_kwargs=fixed_params['EARLY_STOPPING_KWARGS'],
        alpha_epoch_anneal=fixed_params['ALPHA_EPOCH_ANNEAL'],
        pretraining_epochs=PRE_EPOCHS,
        clustering_res=fixed_params['CLUSTERING_RES'],
        eta=fixed_params['ETA'],
    )
    lataq.save(REF_PATH, overwrite=True)
    if model == 'embedcvae':
        lataq_query = EMBEDCVAE.load_query_data(
            adata=target_adata,
            reference_model=REF_PATH,
            labeled_indices=[],
        )
    elif model == 'tranvae':
        lataq_query = TRANVAE.load_query_data(
            adata=target_adata,
            reference_model=REF_PATH,
            labeled_indices=[],
        )
    lataq_query.train(
            n_epochs=EPOCHS,
            early_stopping_kwargs=fixed_params['EARLY_STOPPING_KWARGS'],
            alpha_epoch_anneal=fixed_params['ALPHA_EPOCH_ANNEAL'],
            pretraining_epochs=PRE_EPOCHS,
            clustering_res=fixed_params['CLUSTERING_RES'],
            eta=fixed_params['ETA'],
        )
    
    results_dict = lataq_query.classify(
            adata.X, 
            adata.obs[condition_key], 
            metric=loss_metric
        )
    for i in range(len(cell_type_key)):
        preds = results_dict[cell_type_key[i]]['preds']
        probs = results_dict[cell_type_key[i]]['probs']
        classification_df = pd.DataFrame(
            classification_report(
                y_true=adata.obs[cell_type_key[i]], 
                y_pred=preds,
                output_dict=True
            )
        ).transpose()