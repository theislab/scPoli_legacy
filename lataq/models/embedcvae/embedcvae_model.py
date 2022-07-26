from typing import Optional, Union

import torch
import pickle
from pathlib import PurePath
from scipy.sparse import issparse
from anndata import AnnData
from scarches.models.base._utils import _validate_var_names

from .. import LATAQ
from .._utils import subsample_conditions
from .embedcvae import EmbedCVAE


class EMBEDCVAE(LATAQ):
    """Model for EmbedCVAE, inherits LATAQ.

    Parameters
    ----------
    inject_condition
    """

    def __init__(
        self,
        adata: AnnData,
        condition_key: str = None,
        conditions: Optional[list] = None,
        cell_type_keys: Optional[list] = None,
        cell_types: Optional[dict] = None,
        unknown_ct_names: Optional[list] = None,
        labeled_indices: Optional[list] = None,
        landmarks_labeled: Optional[dict] = None,
        landmarks_unlabeled: Optional[dict] = None,
        hidden_layer_sizes: list = [256, 64],
        latent_dim: int = 10,
        dr_rate: float = 0.05,
        use_mmd: bool = False,
        mmd_on: str = "z",
        mmd_boundary: Optional[int] = None,
        recon_loss: Optional[str] = "nb",
        beta: float = 1,
        use_bn: bool = False,
        use_ln: bool = True,
        inject_condition: list = ["decoder"],
        embedding_dim: int = 10,
        max_norm: int = None,
    ):
        super(EMBEDCVAE, self).__init__(
            adata,
            condition_key,
            conditions,
            cell_type_keys,
            cell_types,
            unknown_ct_names,
            labeled_indices,
            landmarks_labeled,
            landmarks_unlabeled,
            hidden_layer_sizes,
            latent_dim,
            dr_rate,
            use_mmd,
            mmd_on,
            mmd_boundary,
            recon_loss,
            beta,
            use_bn,
            use_ln,
        )
        self.inject_condition_ = inject_condition
        self.embedding_dim_ = embedding_dim
        self.max_norm_ = max_norm
        self.model = EmbedCVAE(
            input_dim=self.input_dim_,
            conditions=self.conditions_,
            cell_types=self.model_cell_types,
            inject_condition=self.inject_condition_,
            embedding_dim=self.embedding_dim_,
            unknown_ct_names=self.unknown_ct_names_,
            landmarks_labeled=self.landmarks_labeled_,
            landmarks_unlabeled=self.landmarks_unlabeled_,
            hidden_layer_sizes=self.hidden_layer_sizes_,
            latent_dim=self.latent_dim_,
            dr_rate=self.dr_rate_,
            # use_mmd=self.use_mmd_,
            # mmd_on=self.mmd_on_,
            # mmd_boundary=self.mmd_boundary_,
            recon_loss=self.recon_loss_,
            beta=self.beta_,
            use_bn=self.use_bn_,
            use_ln=self.use_ln_,
            max_norm=self.max_norm_,
        )
        if self.landmarks_labeled_["mean"] is not None:
            self.landmarks_labeled_["mean"] = self.landmarks_labeled_["mean"].to(
                next(self.model.parameters()).device
            )
            self.landmarks_labeled_["cov"] = self.landmarks_labeled_["cov"].to(
                next(self.model.parameters()).device
            )
        if self.landmarks_unlabeled_["mean"] is not None:
            self.landmarks_unlabeled_["mean"] = self.landmarks_unlabeled_["mean"].to(
                next(self.model.parameters()).device
            )

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            "condition_key": dct["condition_key_"],
            "conditions": dct["conditions_"],
            "cell_type_keys": dct["cell_type_keys_"],
            "cell_types": dct["cell_types_"],
            "labeled_indices": dct["labeled_indices_"],
            "landmarks_labeled": dct["landmarks_labeled_"],
            "landmarks_unlabeled": dct["landmarks_unlabeled_"],
            "hidden_layer_sizes": dct["hidden_layer_sizes_"],
            "latent_dim": dct["latent_dim_"],
            "dr_rate": dct["dr_rate_"],
            "use_mmd": dct["use_mmd_"],
            "mmd_on": dct["mmd_on_"],
            "mmd_boundary": dct["mmd_boundary_"],
            "recon_loss": dct["recon_loss_"],
            "beta": dct["beta_"],
            "use_bn": dct["use_bn_"],
            "use_ln": dct["use_ln_"],
            "embedding_dim": dct["embedding_dim_"],
            "inject_condition": dct["inject_condition_"],
        }

        return init_params

    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, "TRVAE"],
        labeled_indices: Optional[list] = None,
        unknown_ct_names: Optional[list] = None,
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
            adata = _validate_var_names(adata, var_names)
        else:
            attr_dict = reference_model._get_public_attributes()
            model_state_dict = reference_model.model.state_dict()
        init_params = cls._get_init_params_from_dict(attr_dict)

        conditions = init_params["conditions"]
        n_reference_conditions = len(conditions)
        condition_key = init_params["condition_key"]

        new_conditions = []
        adata_conditions = adata.obs[condition_key].unique().tolist()
        # Check if new conditions are already known
        for item in adata_conditions:
            if item not in conditions:
                new_conditions.append(item)

        # Add new conditions to overall conditions
        for condition in new_conditions:
            conditions.append(condition)

        cell_types = init_params["cell_types"]
        cell_type_keys = init_params["cell_type_keys"]

        # Check for cell types in new adata
        adata_cell_types = dict()
        for cell_type_key in cell_type_keys:
            uniq_cts = adata.obs[cell_type_key][labeled_indices].unique().tolist()
            for ct in uniq_cts:
                if ct in adata_cell_types:
                    adata_cell_types[ct].append(cell_type_key)
                else:
                    adata_cell_types[ct] = [cell_type_key]

        if unknown_ct_names is not None:
            for unknown_ct in unknown_ct_names:
                if unknown_ct in adata_cell_types:
                    del adata_cell_types[unknown_ct]

        # Check if new conditions are already known and if not add them
        for key in adata_cell_types:
            if key not in cell_types:
                cell_types[key] = adata_cell_types[key]

        if remove_dropout:
            init_params["dr_rate"] = 0.0

        init_params["labeled_indices"] = labeled_indices
        init_params["unknown_ct_names"] = unknown_ct_names
        new_model = cls(adata, **init_params)
        new_model.model.n_reference_conditions = n_reference_conditions
        print(new_model.model.n_reference_conditions)
        new_model._load_expand_params_from_dict(model_state_dict)

        if freeze:
            new_model.model.freeze = True
            for name, p in new_model.model.named_parameters():
                p.requires_grad = False
                if "embedding" in name:
                    p.requires_grad = True
                if "theta" in name:
                    p.requires_grad = True
                if freeze_expression:
                    if "cond_L.weight" in name:
                        p.requires_grad = False
                else:
                    if "L0" in name or "N0" in name:
                        p.requires_grad = False

        return new_model

    def _load_expand_params_from_dict(self, state_dict):
        load_state_dict = state_dict.copy()

        device = next(self.model.parameters()).device

        new_state_dict = self.model.state_dict()
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new embedding in dictionary
            elif key == "embedding.weight":
                load_ten = load_ten.to(device)
                dim_diff = new_ten.size()[0] - load_ten.size()[0]
                fixed_ten = torch.cat([load_ten, new_ten[-dim_diff:, ...]], dim=0)
                load_state_dict[key] = fixed_ten
            else:
                load_ten = load_ten.to(device)
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                load_state_dict[key] = fixed_ten

        self.model.load_state_dict(load_state_dict)

    @classmethod
    def zero_shot_surgery(cls, adata, model_path, force_cuda=False, copy=False, subsample=1.):
        assert subsample > 0. and subsample <= 1.

        if copy:
            adata = adata.copy()

        with open(PurePath(model_path) / "attr.pkl", "rb") as handle:
            attr_dict = pickle.load(handle)

        ref_conditions = attr_dict["conditions_"]
        condition_key = attr_dict["condition_key_"]

        if subsample < 1.:
            adata = subsample_conditions(adata, condition_key, subsample)

        original_key = "_original_" + condition_key
        adata.obs[original_key] = adata.obs[condition_key].copy()

        adata.strings_to_categoricals()

        original_cats = adata.obs[condition_key].unique()

        adata.obs[condition_key] = adata.obs[condition_key].cat.rename_categories(ref_conditions[:len(original_cats)])

        ref_model = cls.load(model_path, adata)
        if force_cuda:
            ref_model.model = ref_model.model.cuda()

        device = next(ref_model.model.parameters()).device
        print("Device", device)

        rename_cats = {}

        for cat in original_cats:
            cat_mask = adata.obs[original_key] == cat
            X = adata.X[cat_mask]
            print("Processing original category:", cat, "n_obs:", X.shape[0])
            if issparse(X):
                X = X.toarray()
            X = torch.tensor(X, device=device)
            sizefactor = X.sum(-1)
            c = torch.zeros(X.shape[0], device=device, dtype=int)

            min_loss = None
            for ref_cat, ref_cat_val in ref_model.model.condition_encoder.items():
                print("  processing", ref_cat)
                c[:] = ref_cat_val
                _, recon_loss, _, _ = ref_model.model.forward(x=X, batch=c, sizefactor=sizefactor)
                if min_loss is None:
                    min_loss = recon_loss
                    rename_cats[cat] = ref_cat
                else:
                    if recon_loss < min_loss:
                        min_loss = recon_loss
                        rename_cats[cat] = ref_cat

        map_cats = adata.obs[original_key].map(rename_cats).astype("category")

        adata.obs[condition_key] = map_cats
        ref_model.adata.obs[condition_key] = map_cats

        return ref_model, rename_cats

    @classmethod
    def one_shot_surgery(
        cls,
        adata,
        model_path,
        force_cuda=False,
        copy=False,
        subsample=1.,
        pretrain=0,
        **kwargs
    ):
        assert subsample > 0. and subsample <= 1.

        if copy:
            adata = adata.copy()

        ref_model, rename_cats = cls.zero_shot_surgery(adata, model_path, force_cuda=force_cuda, copy=False)

        cond_key = ref_model.condition_key_
        adata.obs[cond_key] = adata.obs["_original_" + cond_key]

        if subsample < 1.:
            adata = subsample_conditions(adata, cond_key, subsample)

        query_model = cls.load_query_data(adata, ref_model, **kwargs)

        cond_enc = query_model.model.condition_encoder

        to_set = [cond_enc[cat] for cat in rename_cats]
        to_get = [cond_enc[cat] for cat in rename_cats.values()]

        query_model.model.embedding.weight.data[to_set] = query_model.model.embedding.weight.data[to_get]

        if pretrain > 0:
            query_model.train(n_epochs=pretrain, pretraining_epochs=pretrain)

        return query_model
