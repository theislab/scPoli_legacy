import torch
import numpy as np

from anndata import AnnData
from typing import Optional, Union

from scarches.models.base._utils import _validate_var_names
from scarches.models.base._base import BaseMixin

from .tranvae import tranVAE
from tranvae.trainers.supervised import tranVAETrainer


class TRANVAE(BaseMixin):
    """Model for scArches class. This class contains the implementation of Conditional Variational Auto-encoder.

       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z)  size.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
       use_mmd: Boolean
            If 'True' an additional MMD loss will be calculated on the latent dim. 'z' or the first decoder layer 'y'.
       mmd_on: String
            Choose on which layer MMD loss will be calculated on if 'use_mmd=True': 'z' for latent dim or 'y' for first
            decoder layer.
       mmd_boundary: Integer or None
            Choose on how many conditions the MMD loss should be calculated on. If 'None' MMD will be calculated on all
            conditions.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
       beta: Float
            Scaling Factor for MMD loss
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
    """
    def __init__(
        self,
        adata: AnnData,
        condition_key: str = None,
        conditions: Optional[list] = None,
        cell_type_key: str = None,
        cell_types: Optional[list] = None,
        labeled_indices: Optional[list] = None,
        n_clusters: Optional[int] = None,
        clustering: Optional[str] = None,
        use_unlabeled_loss: Optional[bool] = True,
        landmarks_labeled: Optional[list] = None,
        landmarks_normalize: Optional[list] = None,
        landmarks_unlabeled: Optional[np.ndarray] = None,
        hidden_layer_sizes: list = [256, 64],
        latent_dim: int = 10,
        dr_rate: float = 0.05,
        use_mmd: bool = True,
        mmd_on: str = 'z',
        mmd_boundary: Optional[int] = None,
        recon_loss: Optional[str] = 'nb',
        beta: float = 1,
        use_bn: bool = False,
        use_ln: bool = True,
    ):
        self.adata = adata

        self.condition_key_ = condition_key
        self.cell_type_key_ = cell_type_key

        if labeled_indices is None:
            self.labeled_indices_ = range(len(adata))
        else:
            self.labeled_indices_ = labeled_indices

        if conditions is None:
            if condition_key is not None:
                self.conditions_ = adata.obs[condition_key].unique().tolist()
            else:
                self.conditions_ = []
        else:
            self.conditions_ = conditions

        if cell_types is None:
            if cell_type_key is not None:
                self.cell_types_ = adata.obs[cell_type_key][self.labeled_indices_].unique().tolist()
            else:
                self.cell_types_ = []
        else:
            self.cell_types_ = cell_types

        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.latent_dim_ = latent_dim
        self.dr_rate_ = dr_rate
        self.use_mmd_ = use_mmd
        self.mmd_on_ = mmd_on
        self.mmd_boundary_ = mmd_boundary
        self.recon_loss_ = recon_loss
        self.beta_ = beta
        self.use_bn_ = use_bn
        self.use_ln_ = use_ln

        self.input_dim_ = adata.n_vars
        self.landmarks_labeled_ = landmarks_labeled
        self.landmarks_normalize_ = landmarks_normalize
        self.landmarks_unlabeled_ = landmarks_unlabeled

        self.model = tranVAE(
            input_dim=self.input_dim_,
            conditions=self.conditions_,
            cell_types=self.cell_types_,
            landmarks_labeled=self.landmarks_labeled_,
            landmarks_normalize=self.landmarks_normalize_,
            landmarks_unlabeled=self.landmarks_unlabeled_,
            hidden_layer_sizes=self.hidden_layer_sizes_,
            latent_dim=self.latent_dim_,
            dr_rate=self.dr_rate_,
            use_mmd=self.use_mmd_,
            mmd_on=self.mmd_on_,
            mmd_boundary=self.mmd_boundary_,
            recon_loss=self.recon_loss_,
            beta=self.beta_,
            use_bn=self.use_bn_,
            use_ln=self.use_ln_,
        )

        self.n_clusters_ = n_clusters
        self.clustering_ = clustering
        self.use_unlabeled_loss_ = use_unlabeled_loss
        self.is_trained_ = False

        self.trainer = None

    def train(
        self,
        n_epochs: int = 400,
        lr: float = 1e-3,
        eps: float = 0.01,
        **kwargs
    ):
        """Train the model.

           Parameters
           ----------
           n_epochs
                Number of epochs for training the model.
           lr
                Learning rate for training the model.
           eps
                torch.optim.Adam eps parameter
           kwargs
                kwargs for the TranVAE trainer.
        """
        self.trainer = tranVAETrainer(
            self.model,
            self.adata,
            n_clusters=self.n_clusters_,
            clustering=self.clustering_,
            use_unlabeled_loss=self.use_unlabeled_loss_,
            labeled_indices=self.labeled_indices_,
            condition_key=self.condition_key_,
            cell_type_key=self.cell_type_key_,
            **kwargs)
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True
        self.landmarks_labeled_ = self.model.landmarks_labeled
        self.landmarks_normalize_ = \
            self.model.landmarks_normalize.cpu().detach().numpy() if self.model.landmarks_normalize is not None else None
        self.landmarks_unlabeled_ = \
            self.model.landmarks_unlabeled.cpu().numpy() if self.model.landmarks_unlabeled is not None else None

    def get_latent(
        self,
        x: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        mean: bool = False
    ):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
           data.

           Parameters
           ----------
           x
                Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
                If None, then `self.adata.X` is used.
           c
                `numpy nd-array` of original (unencoded) desired labels for each sample.
           mean
                return mean instead of random sample from the latent space

           Returns
           -------
                Returns array containing latent space encoding of 'x'.
        """
        device = next(self.model.parameters()).device
        if x is None and c is None:
            x = self.adata.X
            if self.conditions_ is not None:
                c = self.adata.obs[self.condition_key_]

        if c is not None:
            c = np.asarray(c)
            if not set(c).issubset(self.conditions_):
                raise ValueError("Incorrect conditions")
            labels = np.zeros(c.shape[0])
            for condition, label in self.model.condition_encoder.items():
                labels[c == condition] = label
            c = torch.tensor(labels, device=device)

        x = torch.tensor(x, device=device)

        latents = []
        indices = torch.arange(x.size(0), device=device)
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = self.model.get_latent(x[batch,:], c[batch], mean)
            latents += [latent.cpu().detach()]

        return np.array(torch.cat(latents))

    def classify(
            self,
            x: Optional[np.ndarray] = None,
            c: Optional[np.ndarray] = None,
            landmark=False,
            version='prob',
    ):
        device = next(self.model.parameters()).device
        if not landmark:
            if x is None:
                x = self.adata.X
                if self.conditions_ is not None:
                    c = self.adata.obs[self.condition_key_]

            if c is not None:
                c = np.asarray(c)
                if not set(c).issubset(self.conditions_):
                    raise ValueError("Incorrect conditions")
                labels = np.zeros(c.shape[0])
                for condition, label in self.model.condition_encoder.items():
                    labels[c == condition] = label
                c = torch.tensor(labels, device=device)

        x = torch.tensor(x, device=device)

        preds = []
        probs = []
        indices = torch.arange(x.size(0), device=device)
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            if landmark:
                pred, prob = self.model.classify(x[batch, :], landmark=landmark, version=version)
            else:
                pred, prob = self.model.classify(x[batch, :], c[batch], landmark=landmark, version=version)
            preds += [pred.cpu().detach()]
            probs += [prob.cpu().detach()]

        full_pred = np.array(torch.cat(preds))
        inv_ct_encoder = {v: k for k, v in self.model.cell_type_encoder.items()}
        full_pred_names = []

        for pred in full_pred:
            if landmark:
                full_pred_names.append(inv_ct_encoder[pred] + ' Landmark')
            else:
                full_pred_names.append(inv_ct_encoder[pred])

        full_prob = np.array(torch.cat(probs))

        return np.array(full_pred_names), full_prob

    def check_for_unseen(self):
        if self.model.landmarks_unlabeled is not None:
            pred, prob = self.model.check_for_unseen()
            full_prob = prob.detach().cpu().numpy()
            full_pred = pred.detach().cpu().numpy()
            inv_ct_encoder = {v: k for k, v in self.model.cell_type_encoder.items()}
            full_pred_names = []
            for idx, pred in enumerate(full_pred):
                if full_prob[idx]:
                    full_pred_names.append(inv_ct_encoder[pred])
            return np.array(full_pred_names), full_prob
        else:
            print("There are no unlabeled Landmarks in the model.")
            return None, None

    def get_landmarks_info(self):
        landmarks_l = None
        for landmark in self.landmarks_labeled_:
            mean = landmark.mean.detach().cpu().numpy()
            mean = np.expand_dims(mean, axis=0)
            landmarks_l = np.concatenate((landmarks_l, mean)) if landmarks_l is not None else mean

        l_pred, l_prob = self.classify(landmarks_l, landmark=True)

        landmarks_u = self.landmarks_unlabeled_
        u_pred, u_prob = self.classify(landmarks_u, landmark=True)

        x_info = np.concatenate((landmarks_l, landmarks_u))
        label_info = np.concatenate((l_pred, u_pred))
        prob_info = np.concatenate((l_prob, u_prob))
        batch_info = np.array((landmarks_l.shape[0] * ['Landmark Labeled'] +
                               landmarks_u.shape[0] * ['Landmark Unlabeled']))
        return x_info, label_info, batch_info, prob_info

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            'condition_key': dct['condition_key_'],
            'conditions': dct['conditions_'],
            'cell_type_key': dct['cell_type_key_'],
            'cell_types': dct['cell_types_'],
            'labeled_indices': dct['labeled_indices_'],
            'n_clusters': dct['n_clusters_'],
            'clustering': dct['clustering_'],
            'use_unlabeled_loss': dct['use_unlabeled_loss_'],
            'landmarks_labeled': dct['landmarks_labeled_'],
            'landmarks_normalize': dct['landmarks_normalize_'],
            'landmarks_unlabeled': dct['landmarks_unlabeled_'],
            'hidden_layer_sizes': dct['hidden_layer_sizes_'],
            'latent_dim': dct['latent_dim_'],
            'dr_rate': dct['dr_rate_'],
            'use_mmd': dct['use_mmd_'],
            'mmd_on': dct['mmd_on_'],
            'mmd_boundary': dct['mmd_boundary_'],
            'recon_loss': dct['recon_loss_'],
            'beta': dct['beta_'],
            'use_bn': dct['use_bn_'],
            'use_ln': dct['use_ln_'],
        }

        return init_params

    @classmethod
    def _validate_adata(cls, adata, dct):
        if adata.n_vars != dct['input_dim_']:
            raise ValueError("Incorrect var dimension")

        adata_conditions = adata.obs[dct['condition_key_']].unique().tolist()
        if not set(adata_conditions).issubset(dct['conditions_']):
            raise ValueError("Incorrect conditions")

    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, 'TRVAE'],
        labeled_indices: Optional[list] = None,
        n_clusters: Optional[int] = None,
        clustering: Optional[str] = None,
        use_unlabeled_loss: Optional[bool] = True,
        freeze: bool = True,
        freeze_expression: bool = True,
        remove_dropout: bool = True,
    ):
        """Transfer Learning function for new data. Uses old trained model and expands it for new conditions.

           Parameters
           ----------
           adata
                Query anndata object.
           reference_model
                TRVAE model to expand or a path to TRVAE model folder.
           freeze: Boolean
                If 'True' freezes every part of the network except the first layers of encoder/decoder.
           freeze_expression: Boolean
                If 'True' freeze every weight in first layers except the condition weights.
           remove_dropout: Boolean
                If 'True' remove Dropout for Transfer Learning.

           Returns
           -------
           new_model: trVAE
                New TRVAE model to train on query data.
        """
        if isinstance(reference_model, str):
            attr_dict, model_state_dict, var_names = cls._load_params(reference_model)
            _validate_var_names(adata, var_names)
        else:
            attr_dict = reference_model._get_public_attributes()
            model_state_dict = reference_model.model.state_dict()
        init_params = cls._get_init_params_from_dict(attr_dict)

        conditions = init_params['conditions']
        condition_key = init_params['condition_key']

        new_conditions = []
        adata_conditions = adata.obs[condition_key].unique().tolist()
        # Check if new conditions are already known
        for item in adata_conditions:
            if item not in conditions:
                new_conditions.append(item)

        # Add new conditions to overall conditions
        for condition in new_conditions:
            conditions.append(condition)

        cell_types = init_params['cell_types']
        cell_type_key = init_params['cell_type_key']
        new_cell_types = []
        adata_cell_types = adata.obs[cell_type_key][labeled_indices].unique().tolist()
        # Check if new conditions are already known
        for item in adata_cell_types:
            if item not in cell_types:
                new_cell_types.append(item)

        # Add new conditions to overall conditions
        for cell_type in new_cell_types:
            cell_types.append(cell_type)

        if remove_dropout:
            init_params['dr_rate'] = 0.0

        init_params['n_clusters'] = n_clusters
        init_params['clustering'] = clustering
        init_params['use_unlabeled_loss'] = use_unlabeled_loss
        init_params['labeled_indices'] = labeled_indices

        new_model = cls(adata, **init_params)
        new_model._load_expand_params_from_dict(model_state_dict)

        if freeze:
            for name, p in new_model.model.named_parameters():
                p.requires_grad = False
                if 'theta' in name:
                    p.requires_grad = True
                if freeze_expression:
                    if 'cond_L.weight' in name:
                        p.requires_grad = True
                else:
                    if "L0" in name or "N0" in name:
                        p.requires_grad = True

        return new_model
