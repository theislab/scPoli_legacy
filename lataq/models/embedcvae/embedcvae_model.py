from typing import Optional, Union

import torch
from anndata import AnnData
from scarches.models.base._utils import _validate_var_names

from .. import LATAQ
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
