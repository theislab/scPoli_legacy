import pytest

import scanpy as sc
from scarches.dataset.trvae.data_handling import remove_sparsity

from lataq.models import EMBEDCVAE, TRANVAE
from lataq_reproduce.utils import label_encoder
from lataq_reproduce.exp_dict import EXPERIMENT_INFO

@pytest.fixture
def fixed_params():
    return {
        'EPOCHS': 50,
        'N_PRE_EPOCHS': 40,
        'DATA_DIR': '../data'
        'DATA': 'test_data.h5ad',
        'CONDITION_KEY':,
        'CT_KEY':,
        'REFERENCE':,
        'QUERY':,
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

def test_fatal(fixed_params, model, loss_metric,):
    adata = sc.read(f'{fixed_params['DATA_DIR']}/{fixed_params['DATA']}')
    condition_key = fixed_params['CONDITION_KEY']
    cell_type_key = fixed_params['CT_KEY']
    reference = fixed_params['REFERENCE']
    query = fixed_params['QUERY']

    EPOCHS = fixed_params['EPOCHS']
    PRE_EPOCHS = fixed_params['N_PRE_EPOCHS']

    adata = remove_sparsity(adata)
    source_adata = adata[adata.obs.study.isin(reference)].copy()
    target_adata = adata[adata.obs.study.isin(query)].copy()

    #hyperbolic_log1p = False
    
    #if loss_metric == 'hyperbolic_log1p':
    #    loss_metric = 'hyperbolic'
    #    hyperbolic_log1p=True

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
        eta=eta,
    )
    lataq.save(REF_PATH, overwrite=True)
    if model == 'embedcvae':
        lataq_query = EMBEDCVAE.load_query_data(
            adata=target_adata,
            reference_model=REF_PATH,
            labeled_indices=[],
        )
    elif model == 'tranvae':
        lataq_query = lataq.load_query_data(
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
    
    logging.info('Computing metrics')
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
    


class HelperEstimator(HelperEstimatorBase):

    estimator: Union[EstimatorKerasMulticellCell, EstimatorKerasMulticellType]
    data: Union[DistributedStoreSingleFeatureSpace]
    model_type: str
    tc: TopologyContainer

    def __init__(self, model):
        self._adata_ids = AdataIdsSfaira()
        self.model = model

    def init_topology(self):
        topology = TOPOLOGY_MULTICELL_MODEL.copy()
        topology["model_type"] = self.model
        tc = TopologyContainer(topology=topology, topology_id="0.0.1")
        self.model_type = tc.model_type
        self.tc = tc

    def init_estimator(self):
        self.estimator = EstimatorKerasMulticellType(
            data=self.data,
            model_dir=DIR_TEMP,
            cache_path=DIR_TEMP,
            model_id="testid",
            model_topology=self.tc
        )

    def basic_estimator_test(self):
        self.estimator.init_model()
        self.estimator.train(
            optimizer="adam",
            lr=0.005,
            epochs=2,
            batch_size=32,
            max_steps_per_epoch=1,
            validation_split=0.1,
            test_split=0.1,
            validation_batch_size=2,
            max_validation_steps=1,
        )
        _ = self.estimator.evaluate()
        _ = self.estimator.predict()

    def test_for_fatal(self):
        self.init_topology()
        self.load_store(organism="human")
        self.init_estimator()
        self.basic_estimator_test()


@pytest.mark.parametrize("model", ["typev1"])
def test_for_fatal_base(model):
    test_estim = HelperEstimator(model=model)
    test_estim.test_for_fatal()