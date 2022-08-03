from typing import Optional, Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from scarches.models.base._base import BaseMixin
from .lataq import EmbedCVAE
from lataq.trainers import LATAQtrainer


class LATAQ(BaseMixin):
    """Model for LATAQ class. This class contains the methods and functionalities for label transfer and landmark training.

    Parameters
    ----------
    adata: : `~anndata.AnnData`
        Annotated data matrix.
    condition_key: String
        column name of conditions in `adata.obs` data frame.
    conditions: List
        List of Condition names that the used data will contain to get the right encoding when used after reloading.
    cell_type_keys: List
        List of obs columns to use as cell type annotation for landmarks.
    cell_types: Dictionary
        Dictionary of cell types. Keys are cell types and values are cell_type_keys. Needed for surgery.
    unknown_ct_names: List
        List of strings with the names of cell clusters to be ignored for landmarks computation.
    labeled_indices: List
        List of integers with the indices of the labeled cells.
    landmarks_labeled: Dictionary
        Dictionary with keys mean, cov and the respective mean or covariate matrices for landmarks.
    landmarks_unlabeled: Dictionary
        Dictionary with keys mean and the respective mean for unlabeled landmarks.
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
        inject_condition: Optional[list] = ['encoder', 'decoder'],
        cell_type_keys: Optional[list] = None,
        cell_types: Optional[dict] = None,
        unknown_ct_names: Optional[list] = None,
        labeled_indices: Optional[list] = None,
        landmarks_labeled: Optional[dict] = None,
        landmarks_unlabeled: Optional[dict] = None,
        hidden_layer_sizes: list = [256, 64],
        latent_dim: int = 10,
        embedding_dim: int = 10,
        dr_rate: float = 0.05,
        use_mmd: bool = False,
        mmd_on: str = "z",
        mmd_boundary: Optional[int] = None,
        recon_loss: Optional[str] = "nb",
        beta: float = 1,
        use_bn: bool = False,
        use_ln: bool = True,
    ):

        # gather data information
        self.adata = adata
        self.condition_key_ = condition_key
        self.cell_type_keys_ = cell_type_keys
        if unknown_ct_names is not None and type(unknown_ct_names) is not list:
            raise TypeError(
                f"Parameter 'unknown_ct_names' has to be list not {type(unknown_ct_names)}"
            )
        self.unknown_ct_names_ = unknown_ct_names

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

        # Gather all cell type information
        if cell_types is None:
            if cell_type_keys is not None:
                self.cell_types_ = dict()
                for cell_type_key in cell_type_keys:
                    uniq_cts = (
                        adata.obs[cell_type_key][self.labeled_indices_]
                        .unique()
                        .tolist()
                    )
                    for ct in uniq_cts:
                        if ct in self.cell_types_:
                            self.cell_types_[ct].append(cell_type_key)
                        else:
                            self.cell_types_[ct] = [cell_type_key]
            else:
                self.cell_types_ = dict()
        else:
            self.cell_types_ = cell_types

        if self.unknown_ct_names_ is not None:
            for unknown_ct in self.unknown_ct_names_:
                if unknown_ct in self.cell_types_:
                    del self.cell_types_[unknown_ct]

        # store model parameters
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
        self.inject_condition_ = inject_condition
        self.embedding_dim_ = embedding_dim

        self.input_dim_ = adata.n_vars
        self.landmarks_labeled_ = (
            {"mean": None, "cov": None}
            if landmarks_labeled is None
            else landmarks_labeled
        )
        self.landmarks_unlabeled_ = (
            {
                "mean": None,
            }
            if landmarks_unlabeled is None
            else landmarks_unlabeled
        )

        self.model_cell_types = list(self.cell_types_.keys())
        self.is_trained_ = False
        self.trainer = None
        
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
    
    def train(self, n_epochs: int = 400, lr: float = 1e-3, eps: float = 0.01, **kwargs):
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
        self.trainer = LATAQtrainer(
            self.model,
            self.adata,
            labeled_indices=self.labeled_indices_,
            condition_key=self.condition_key_,
            cell_type_keys=self.cell_type_keys_,
            **kwargs,
        )
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True
        self.landmarks_labeled_ = self.model.landmarks_labeled
        self.landmarks_unlabeled_ = self.model.landmarks_unlabeled

    def get_latent(
        self,
        x: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        mean: bool = False,
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
            c = torch.tensor(labels, device="cpu")

        x = torch.tensor(x, device="cpu")

        latents = []
        # batch the latent transformation process
        indices = torch.arange(x.size(0), device="cpu")
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = self.model.get_latent(
                x[batch, :].to(device), c[batch].to(device), mean
            )
            latents += [latent.cpu().detach()]

        latents = torch.cat(latents)
        return np.array(latents)

    def classify(
        self,
        x: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        landmark=False,
        get_prob=True,
        threshold=0,
    ):
        """
        Classifies unlabeled cells using the landmarks obtained during training.
        Data handling before call to model's classify method.

        x:  np.ndarray
            Features to be classified. If None the stored
            model's adata is used.
        c: np.ndarray
            Condition vector.
        landmark:
            Boolean whether to classify the gene features or landmarks stored
            stored in the model.
        threshold:
            Threshold to use on the class probabilities to detect novel cell types,
            or mark unknown cells.


        """
        device = next(self.model.parameters()).device

        if not landmark:
            # get the gene features from stored adata
            if x is None:
                x = self.adata.X
                if self.conditions_ is not None:
                    c = self.adata.obs[self.condition_key_]
            # get the conditions from passed input
            if c is not None:
                c = np.asarray(c)
                if not set(c).issubset(self.conditions_):
                    raise ValueError("Incorrect conditions")
                labels = np.zeros(c.shape[0])
                for condition, label in self.model.condition_encoder.items():
                    labels[c == condition] = label
                c = torch.tensor(labels, device="cpu")

        x = torch.tensor(x, device="cpu")

        results = dict()
        # loop through hierarchies
        for cell_type_key in self.cell_type_keys_:
            landmarks_idx = list()
            # get indices of different landmarks corresponding to current hierarchy
            for i, key in enumerate(self.cell_types_.keys()):
                if cell_type_key in self.cell_types_[key]:
                    landmarks_idx.append(i)

            landmarks_idx = torch.tensor(landmarks_idx, device=device)

            preds = []
            probs = []
            weighted_distances = []
            indices = torch.arange(x.size(0), device=device)
            subsampled_indices = indices.split(512)
            for batch in subsampled_indices:
                if landmark:  # classify landmarks used for unseen cell type
                    pred, prob, weighted_distance = self.model.classify(
                        x[batch, :].to(device),
                        landmark=landmark,
                        classes_list=landmarks_idx,
                        get_prob=get_prob,
                    )
                else:  # default routine, classify cell by cell
                    pred, prob, weighted_distance = self.model.classify(
                        x[batch, :].to(device),
                        c[batch].to(device),
                        landmark=landmark,
                        classes_list=landmarks_idx,
                        get_prob=get_prob,
                    )
                preds += [pred.cpu().detach()]
                probs += [prob.cpu().detach()]
                weighted_distances += [weighted_distance.cpu().detach()]

            full_pred = np.array(torch.cat(preds))
            full_prob = np.array(torch.cat(probs))
            full_weighted_distances = np.array(torch.cat(weighted_distances))
            inv_ct_encoder = {v: k for k, v in self.model.cell_type_encoder.items()}
            full_pred_names = []

            for idx, pred in enumerate(full_pred):
                if full_prob[idx] > threshold:
                    full_pred_names.append(inv_ct_encoder[pred])
                else:
                    full_pred_names.append(f"nan")

            results[cell_type_key] = {
                "preds": np.array(full_pred_names),
                "probs": full_prob,
                "weighted_distances": full_weighted_distances,
            }

        return results

    def add_new_cell_type(
        self,
        cell_type_name,
        obs_key,
        landmarks,
        x=None,
        c=None,
    ):
        """
        Function used to add new annotation for a novel cell type.

        Parameters
        ----------
        cell_type_name: str
            Name of the new cell type
        obs_key: str
            Obs column key to define the hierarchy level of celltype annotation.
        landmarks: list
            List of indices of the unlabeled landmarks that correspond to the new cell type
        x:  np.ndarray
            Features to be classified. If None the stored
            model's adata is used.
        c: np.ndarray
            Condition vector. If None the stored
            model's condition vector is used.

        Returns
        -------

        """
        # Get latent of model data or input data
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
            c = torch.tensor(labels, device="cpu")
        x = torch.tensor(x, device="cpu")
        latents = []
        indices = torch.arange(x.size(0), device="cpu")
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = self.model.get_latent(
                x[batch, :].to(device), c[batch].to(device), False
            )
            latents += [latent.cpu().detach()]
        latents = torch.cat(latents)

        # get indices of different landmarks corresponding to current hierarchy
        landmarks_idx = list()
        for i, key in enumerate(self.cell_types_.keys()):
            if obs_key in self.cell_types_[key]:
                landmarks_idx.append(i)

        landmarks_idx = torch.tensor(landmarks_idx, device=device)

        # Calculate mean and Cov of new landmark
        self.model.add_new_cell_type(
            latents,
            cell_type_name,
            landmarks,
            landmarks_idx,
        )

        # Update parameters
        self.landmarks_labeled_ = self.model.landmarks_labeled
        self.landmarks_unlabeled_ = self.model.landmarks_unlabeled
        self.cell_types_[cell_type_name] = [obs_key]

    def get_landmarks_info(
        self, landmark_set="labeled", threshold=0,
    ):
        """
        Generates anndata file with landmark features and annotations.

        Parameters
        ----------
        cell_type_name: str
            Name of the new cell type
        landmarks: list
            List of indices of the unlabeled landmarks that correspond to the new cell type

        Returns
        -------

        """
        if landmark_set == "labeled":
            landmarks = self.landmarks_labeled_["mean"].detach().cpu().numpy()
            batch_name = "Landmark-Set Labeled"
        elif landmark_set == "unlabeled":
            landmarks = self.landmarks_unlabeled_["mean"].detach().cpu().numpy()
            batch_name = "Landmark-Set Unlabeled"
        else:
            print(
                f"Parameter 'landmark_set' has either to be 'labeled' for labeled landmark set or 'unlabeled' "
                f"for the unlabeled landmark set. But given value was {landmark_set}"
            )
            return
        landmarks_info = sc.AnnData(landmarks)
        landmarks_info.obs[self.condition_key_] = np.array(
            (landmarks.shape[0] * [batch_name])
        )

        results = self.classify(
            landmarks, landmark=True, threshold=threshold
        )
        for cell_type_key in self.cell_type_keys_:
            if landmark_set == "l":
                truth_names = list()
                for key in self.cell_types_.keys():
                    if cell_type_key in self.cell_types_[key]:
                        truth_names.append(key)
                    else:
                        truth_names.append("nan")
            else:
                truth_names = list()
                for i in range(landmarks.shape[0]):
                    truth_names.append(f"{i}")

            landmarks_info.obs[cell_type_key] = np.array(truth_names)
            landmarks_info.obs[cell_type_key + "_pred"] = results[cell_type_key][
                "preds"
            ]
            landmarks_info.obs[cell_type_key + "_prob"] = results[cell_type_key][
                "probs"
            ]
        return landmarks_info

    @classmethod
    def _validate_adata(cls, adata, dct):
        if adata.n_vars != dct["input_dim_"]:
            raise ValueError("Incorrect var dimension")

        adata_conditions = adata.obs[dct["condition_key_"]].unique().tolist()
        if not set(adata_conditions).issubset(dct["conditions_"]):
            raise ValueError("Incorrect conditions")

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
        reference_model: Union[str, "EMBEDCVAE"],
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