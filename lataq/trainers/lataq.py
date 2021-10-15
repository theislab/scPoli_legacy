
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scarches.trainers.trvae._utils import make_dataset
from scarches.trainers.trvae.trainer import Trainer
from sklearn.cluster import KMeans

from ._utils import cov, euclidean_dist, kl_loss, t_dist, target_distribution


class LATAQtrainer(Trainer):
    """
    LATAQ Trainer class. This class contains the implementation of the training routine for LATAQ models

    Parameters
    ----------
    model: TRANVAE, EMBEDCVAE
        PyTorch model to train
    adata: : `~anndata.AnnData`
        Annotated data matrix.
    condition_key: String
        column name of conditions in `adata.obs` data frame.
    cell_type_key: String
        column name of celltypes in `adata.obs` data frame.
    train_frac: Float
        Defines the fraction of data that is used for training and data that is used for validation.
    batch_size: Integer
        Defines the batch size that is used during each Iteration
    n_samples: Integer or None
        Defines how many samples are being used during each epoch. This should only be used if hardware resources
        are limited.
    clip_value: Float
        If the value is greater than 0, all gradients with an higher value will be clipped during training.
    weight decay: Float
        Defines the scaling factor for weight decay in the Adam optimizer.
    alpha_iter_anneal: Integer or None
        If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every iteration until the input
        integer is reached.
    alpha_epoch_anneal: Integer or None
        If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every epoch until the input
        integer is reached.
    use_early_stopping: Boolean
        If 'True' the EarlyStopping class is being used for training to prevent overfitting.
    early_stopping_kwargs: Dict
        Passes custom Earlystopping parameters.
    use_stratified_sampling: Boolean
        If 'True', the sampler tries to load equally distributed batches concerning the conditions in every
        iteration.
    monitor: Boolean
        If `True', the progress of the training will be printed after each epoch.
    n_workers: Integer
        Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
    labeled_indices: list
        List of integers that indicate the annotated data
    pretraining_epochs: Integer
        Number of epochs for pretraining
    clustering: String
        Clustering methodology to use, can be KMeans or Leiden.
    clustering_res: Float
        Clustering resolution to use for leiden clustering. Bigger values result in
        finer clusters.
    n_clusters: Integer
        Number of clusters to set for KMeans algorithm.
    loss_metric: String
        Type of loss to use for landmarks
    unlabeled_weight: Float
        Weight for loss computed including unlabeled samples
    overconfidence_scale: Integer
        Parameter to use in case of hyperbolic loss
    hyperbolic_log1p: Boolean
        Whether to use log1p transform in hyperbolic loss
    eta: Float
        Weight for the landmark loss
    seed: Integer
        Define a specific random seed to get reproducable results.
    """

    def __init__(
        self,
        model,
        adata,
        labeled_indices: list = None,
        pretraining_epochs: int = 0,
        clustering: str = "leiden",
        clustering_res: float = 1,
        n_clusters: int = None,
        loss_metric: str = "dist",
        unlabeled_weight: float = 0.001,
        overconfidence_scale: int = None,
        hyperbolic_log1p: bool = False,
        eta: float = 1,
        **kwargs,
    ):

        super().__init__(model, adata, **kwargs)

        # Only necessary for hyperbolic loss:
        if overconfidence_scale is None:
            self.overconfidence_scale = self.model.latent_dim
        else:
            self.overconfidence_scale = overconfidence_scale
        if hyperbolic_log1p:
            self.log1p_scale = 1
        else:
            self.log1p_scale = 0

        self.loss_metric = loss_metric
        self.eta = eta
        self.clustering = clustering
        self.n_clusters = n_clusters
        self.unlabeled_weight = unlabeled_weight
        self.clustering_res = clustering_res
        self.pretraining_epochs = pretraining_epochs
        self.use_early_stopping_orig = self.use_early_stopping

        self.landmarks_labeled = None  # landmarks labeled cells (means)
        self.landmarks_labeled_cov = None  # landmarks labeled cells (cov)
        self.landmarks_unlabeled = None  # landmarks all cells (means)
        self.best_landmarks_labeled = None  # cache for ES, to use best state
        self.best_landmarks_labeled_cov = None  # cache for ES
        self.best_landmarks_unlabeled = None  # cache for ES
        self.landmark_optim = None  # landmark optimizer
        # Set indices for labeled data
        if labeled_indices is None:
            self.labeled_indices = range(len(adata))
        else:
            self.labeled_indices = labeled_indices
        self.update_labeled_indices(self.labeled_indices)

        # Parse landmarks from model into right format
        if self.model.landmarks_labeled["mean"] is not None:
            self.landmarks_labeled = self.model.landmarks_labeled["mean"]
            self.landmarks_labeled_cov = self.model.landmarks_labeled["cov"]
        if self.landmarks_labeled is not None:
            self.landmarks_labeled = self.landmarks_labeled.to(device=self.device)
            self.landmarks_labeled_cov = self.landmarks_labeled_cov.to(
                device=self.device
            )

    def update_labeled_indices(self, labeled_indices):
        """
        Function to generate a dataset with new labeled indices after init.

        Parameters
        ==========
        labeled_indices: list
            List of integer indices for labeled samples.

        """
        self.labeled_indices = labeled_indices
        self.train_data, self.valid_data = make_dataset(
            self.adata,
            train_frac=self.train_frac,
            condition_key=self.condition_key,
            cell_type_keys=self.cell_type_keys,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
            labeled_indices=self.labeled_indices,
        )

    def get_latent_train(self):
        """
        Function to return the latent representation of the training dataset.

        Returns
        =======
        latent
        """
        latents = []
        indices = torch.arange(self.train_data.data.size(0), device=self.device)
        subsampled_indices = indices.split(self.batch_size)
        for batch in subsampled_indices:
            latent = self.model.get_latent(
                self.train_data.data[batch, :].to(self.device),
                self.train_data.conditions[batch].to(self.device),
            )
            latents += [latent.cpu().detach()]
        latent = torch.cat(latents)
        return latent.to(self.device)

    def initialize_landmarks(self):
        """
        Function that initializes landmarks
        """
        # Compute Latent of whole train data
        latent = self.get_latent_train()

        # Init labeled Landmarks if labeled data existent
        if 1 in self.train_data.labeled_vector.unique().tolist():
            labeled_latent = latent[self.train_data.labeled_vector == 1]
            labeled_cell_types = self.train_data.cell_types[
                self.train_data.labeled_vector == 1, :
            ]  # get cell type annot
            if (
                self.landmarks_labeled is not None
            ):  # checks if model already has initialized landmarks and then initialize new landmarks for new or unseen cell types in query
                with torch.no_grad():
                    if len(self.model.new_landmarks) > 0:
                        for value in self.model.new_landmarks:
                            indices = labeled_cell_types.eq(value).nonzero(
                                as_tuple=False
                            )[:, 0]
                            landmark = labeled_latent[indices].mean(0)
                            landmark_cov = cov(labeled_latent[indices]).unsqueeze(0)
                            self.landmarks_labeled = torch.cat(
                                [self.landmarks_labeled, landmark]
                            )
                            self.landmarks_labeled_cov = torch.cat(
                                [self.landmarks_labeled_cov, landmark_cov]
                            )
            else:  # compute labeled landmarks
                (
                    self.landmarks_labeled,
                    self.landmarks_labeled_cov,
                ) = self.update_labeled_landmarks(
                    latent[self.train_data.labeled_vector == 1],
                    self.train_data.cell_types[self.train_data.labeled_vector == 1, :],
                    None,
                    None,
                )

        # Init unlabeled Landmarks if unlabeled data existent
        # Unknown ct names: list of strings that identify cells to ignore during training
        if (
            0 in self.train_data.labeled_vector.unique().tolist()
            or self.model.unknown_ct_names is not None
        ):
            lat_array = latent.cpu().detach().numpy()

            if self.clustering == "kmeans" and self.n_clusters is not None:
                print(
                    f"\nInitializing unlabeled landmarks with KMeans-Clustering with a given number of"
                    f"{self.n_clusters} clusters."
                )
                k_means = KMeans(n_clusters=self.n_clusters).fit(lat_array)
                k_means_landmarks = torch.tensor(
                    k_means.cluster_centers_, device=self.device
                )

                self.landmarks_unlabeled = [
                    torch.zeros(
                        size=(1, self.model.latent_dim),
                        requires_grad=False,
                        device=self.device,
                    )
                    for _ in range(self.n_clusters)
                ]  # initialize tensor with zeros

                with torch.no_grad():
                    [
                        self.landmarks_unlabeled[i].copy_(k_means_landmarks[i, :])
                        for i in range(k_means_landmarks.shape[0])
                    ]
                    # replace zeros with the kmeans centroids
            else:
                if self.clustering == "kmeans" and self.n_clusters is None:
                    print(
                        f"\nInitializing unlabeled landmarks with Leiden-Clustering because no value for the"
                        f"number of clusters was given."
                    )
                else:
                    print(
                        f"\nInitializing unlabeled landmarks with Leiden-Clustering with an unknown number of "
                        f"clusters."
                    )
                lat_adata = sc.AnnData(lat_array)
                sc.pp.neighbors(lat_adata)
                sc.tl.leiden(lat_adata, resolution=self.clustering_res)

                features = pd.DataFrame(
                    lat_adata.X, index=np.arange(0, lat_adata.shape[0])
                )
                group = pd.Series(
                    np.asarray(lat_adata.obs["leiden"], dtype=int),
                    index=np.arange(0, lat_adata.shape[0]),
                    name="cluster",
                )
                merged_df = pd.concat([features, group], axis=1)
                cluster_centers = np.asarray(merged_df.groupby("cluster").mean())

                self.n_clusters = cluster_centers.shape[0]
                print(f"Leiden Clustering succesful. Found {self.n_clusters} clusters.")
                leiden_landmarks = torch.tensor(cluster_centers, device=self.device)

                self.landmarks_unlabeled = [
                    torch.zeros(
                        size=(1, self.model.latent_dim),
                        requires_grad=False,
                        device=self.device,
                    )
                    for _ in range(self.n_clusters)
                ]

                with torch.no_grad():
                    [
                        self.landmarks_unlabeled[i].copy_(leiden_landmarks[i, :])
                        for i in range(leiden_landmarks.shape[0])
                    ]

    def on_epoch_begin(self, lr, eps):
        """
        Routine that happens at the beginning of every epoch. Model update step.
        """
        if self.epoch == self.pretraining_epochs:
            self.initialize_landmarks()
            if (
                0 in self.train_data.labeled_vector.unique().tolist()
                or self.model.unknown_ct_names is not None
            ):
                self.landmark_optim = torch.optim.Adam(
                    params=self.landmarks_unlabeled,
                    lr=lr,
                    eps=eps,
                    weight_decay=self.weight_decay,
                )
        if self.epoch < self.pretraining_epochs:
            self.use_early_stopping = False
        if self.use_early_stopping_orig and self.epoch >= self.pretraining_epochs:
            self.use_early_stopping = True
        if self.epoch >= self.pretraining_epochs and self.epoch - 1 == self.best_epoch:
            self.best_landmarks_labeled = self.landmarks_labeled
            self.best_landmarks_labeled_cov = self.landmarks_labeled_cov
            self.best_landmarks_unlabeled = self.landmarks_unlabeled

    def loss(self, total_batch=None):
        latent, recon_loss, kl_loss, mmd_loss = self.model(**total_batch)

        # Calculate classifier loss for labeled/unlabeled data
        label_categories = total_batch["labeled"].unique().tolist()
        unweighted_landmark_loss = torch.tensor(0.0, device=self.device)
        unlabeled_loss = torch.tensor(0.0, device=self.device)
        labeled_loss = torch.tensor(0.0, device=self.device)
        if self.epoch >= self.pretraining_epochs:
            # Calculate landmark loss for all data
            if self.landmarks_unlabeled is not None and self.unlabeled_weight > 0:
                unlabeled_loss, _ = self.landmark_unlabeled_loss(
                    latent,
                    torch.stack(self.landmarks_unlabeled).squeeze(),
                )
                unweighted_landmark_loss = (
                    unweighted_landmark_loss + self.unlabeled_weight * unlabeled_loss
                )

            # Calculate landmark loss for labeled data
            if 1 in label_categories:
                labeled_loss = self.landmark_labeled_loss(
                    latent[total_batch["labeled"] == 1],
                    self.landmarks_labeled,
                    total_batch["celltypes"][total_batch["labeled"] == 1, :],
                )
                unweighted_landmark_loss = unweighted_landmark_loss + labeled_loss

        # Loss addition and Logs
        landmark_loss = self.eta * unweighted_landmark_loss
        trvae_loss = recon_loss + self.calc_alpha_coeff() * kl_loss + mmd_loss
        loss = trvae_loss + landmark_loss
        self.iter_logs["loss"].append(loss.item())
        self.iter_logs["unweighted_loss"].append(
            recon_loss.item()
            + kl_loss.item()
            + mmd_loss.item()
            + unweighted_landmark_loss.item()
        )
        self.iter_logs["trvae_loss"].append(trvae_loss.item())
        if self.epoch >= self.pretraining_epochs:
            self.iter_logs["landmark_loss"].append(landmark_loss.item())
            if 0 in label_categories or self.model.unknown_ct_names is not None:
                self.iter_logs["unlabeled_loss"].append(unlabeled_loss.item())
            if 1 in label_categories:
                self.iter_logs["labeled_loss"].append(labeled_loss.item())
        return loss

    def on_epoch_end(self):
        """
        Routine at the end of each epoch. Landmark update step.
        """
        self.model.eval()

        if self.epoch >= self.pretraining_epochs:
            latent = self.get_latent_train()
            label_categories = self.train_data.labeled_vector.unique().tolist()

            # Update labeled landmark positions
            if 1 in label_categories:
                (
                    self.landmarks_labeled,
                    self.landmarks_labeled_cov,
                ) = self.update_labeled_landmarks(
                    latent[self.train_data.labeled_vector == 1],
                    self.train_data.cell_types[self.train_data.labeled_vector == 1, :],
                    self.landmarks_labeled,
                    self.landmarks_labeled_cov,
                    self.model.new_landmarks,
                )

            # Update unlabeled landmark positions
            if 0 in label_categories or self.model.unknown_ct_names is not None:
                for landmk in self.landmarks_unlabeled:
                    landmk.requires_grad = True
                self.landmark_optim.zero_grad()
                update_loss, args_count = self.landmark_unlabeled_loss(
                    latent,
                    torch.stack(self.landmarks_unlabeled).squeeze(),
                )
                update_loss.backward()
                self.landmark_optim.step()
                for landmk in self.landmarks_unlabeled:
                    landmk.requires_grad = False

        self.model.train()
        super().on_epoch_end()

    def after_loop(self):
        """
        Routine at the end of training. Load best state.
        """
        if self.best_state_dict is not None and self.reload_best:
            self.landmarks_labeled = self.best_landmarks_labeled
            self.landmarks_labeled_cov = self.best_landmarks_labeled_cov
            self.landmarks_unlabeled = self.best_landmarks_unlabeled

        self.model.landmarks_labeled["mean"] = self.landmarks_labeled
        self.model.landmarks_labeled["cov"] = self.landmarks_labeled_cov

        if (
            0 in self.train_data.labeled_vector.unique().tolist()
            or self.model.unknown_ct_names is not None
        ):
            self.model.landmarks_unlabeled["mean"] = torch.stack(
                self.landmarks_unlabeled
            ).squeeze()
        else:
            self.model.landmarks_unlabeled["mean"] = self.landmarks_unlabeled

    def update_labeled_landmarks(
        self, latent, labels, previous_landmarks, previous_landmarks_cov, mask=None
    ):
        """
        Function that updates labeled landmarks.

        Parameters
        ==========
        latent: Tensor
            Latent representation of labeled batch
        labels: Tensor
            Tensor containing cell type information of the batch
        previous_landmarks: Tensor
            Tensor containing the means of the landmarks before update
        previous_landmarks_cov: Tensor
            Tensor containing the covariance matrices of the landmarks before udpate.

        """
        with torch.no_grad():
            unique_labels = torch.unique(labels, sorted=True)
            landmarks_mean = None
            landmarks_cov = None
            for value in range(self.model.n_cell_types):
                if (
                    mask is None or value in mask
                ) and value in unique_labels:  # update the landmark included in mask if there is one
                    indices = labels.eq(value).nonzero(as_tuple=False)[:, 0]
                    landmark = latent[indices, :].mean(0).unsqueeze(0)
                    landmark_cov = cov(latent[indices, :]).unsqueeze(0)
                    landmarks_mean = (
                        torch.cat([landmarks_mean, landmark])
                        if landmarks_mean is not None
                        else landmark
                    )
                    landmarks_cov = (
                        torch.cat([landmarks_cov, landmark_cov])
                        if landmarks_cov is not None
                        else landmark_cov
                    )
                else:  # do not update the landmarks (e.g. during surgery landmarks are fixed)
                    landmark = previous_landmarks[value].unsqueeze(0)
                    landmark_cov = previous_landmarks_cov[value].unsqueeze(0)
                    landmarks_mean = (
                        torch.cat([landmarks_mean, landmark])
                        if landmarks_mean is not None
                        else landmark
                    )
                    landmarks_cov = (
                        torch.cat([landmarks_cov, landmark_cov])
                        if landmarks_cov is not None
                        else landmark_cov
                    )
        return landmarks_mean, landmarks_cov

    def landmark_labeled_loss(self, latent, landmarks, labels):
        """
        Compute the labeled landmark loss. Different losses are included.

        Parameters
        ==========
        latent: Tensor
            Latent representation of labeled batch
        landmarks: Tensor
            Tensor containing the means of the landmarks
        labels: Tensor
            Tensor containing cell type information of the batch
        """
        unique_labels = torch.unique(labels, sorted=True)
        distances = euclidean_dist(latent, landmarks)
        loss = torch.tensor(0.0, device=self.device)

        # If data only contains 'unknown' celltypes
        if unique_labels.tolist() == [-1]:
            return loss

        if self.loss_metric == "dist":
            # Basic euclidean distance loss
            for value in unique_labels:
                if value == -1:
                    continue
                indices = labels.eq(value).nonzero(as_tuple=False)[:, 0]
                label_loss = distances[indices, value].sum(0) / len(indices)
                loss += label_loss

        elif self.loss_metric == "hyperbolic":
            # Transform Landmarks to hyperbolic ideal points
            h_landmarks = F.normalize(landmarks, p=2, dim=1)

            # Transform latent to hyperbolic space and filter out cells with label == -1 which correspond to "unknown"
            transformation_m = (
                (
                    torch.tanh(torch.norm(latent, p=2, dim=1) / 2)
                    / torch.norm(latent, p=2, dim=1)
                )
                .unsqueeze(dim=1)
                .expand(-1, latent.size(1))
            )
            h_latent = transformation_m * latent
            h_latent = h_latent[labels.squeeze(1) != -1, :]

            # Get tensor of corresponding landmarks and filter out cells with label == -1 which correspond to "unknown"
            corr_land = h_landmarks[labels.squeeze(1), :]
            corr_land = corr_land[labels.squeeze(1) != -1, :]

            # Buseman loss
            b_loss = torch.log(
                self.log1p_scale
                + torch.norm(corr_land - h_latent, p=2, dim=1) ** 2
                / (1 - torch.norm(h_latent, p=2, dim=1) ** 2)
            )

            # Overconfidence penalty loss
            overconf_loss = torch.log(1 - torch.norm(h_latent, p=2, dim=1) ** 2)

            # Calculate overall loss by taking mean of each cell
            loss = (b_loss - self.overconfidence_scale * overconf_loss).mean()

        elif self.loss_metric == "overlap":
            # Own idea of cell balls with center at landmark and radius of 95%-quantile
            assert False, "This loss may not work at current state"
            id = torch.eye(len(landmarks), device=self.device)
            truth_id = id[labels]
            quantiles_view = self.landmarks_labeled_cov.unsqueeze(0).expand(
                distances.size(0), distances.size(1)
            )
            overlap = torch.max(
                torch.zeros_like(distances),
                (quantiles_view - distances) / quantiles_view,
            )
            loss = torch.pow(truth_id - overlap, 2).sum(1).mean(0)

        elif self.loss_metric == "seurat":
            # Idea of using seurat distances for loss
            # See https://www.cell.com/cell/pdf/S0092-8674(19)30559-8.pdf
            # assert False, "This loss may not work at current state"
            dists_t = 1 - (distances.T / distances.max(1)[0]).T
            prob = 1 - torch.exp(-dists_t / 4)
            prob = (prob.T / prob.sum(1)).T
            id = torch.eye(len(landmarks), device=self.device)
            truth_id = id[labels]
            loss = torch.pow(truth_id - prob, 2).sum(1).mean(0)

        else:
            assert False, (
                f"'{self.loss_metric}' is not available as a loss function please choose "
                f"between 'dist','t' or 'seurat'!"
            )

        return loss

    def landmark_unlabeled_loss(self, latent, landmarks):
        """
        Compute the unlabeled landmark loss. Different losses are included.

            Parameters
            ==========
            latent: Tensor
                Latent representation of labeled batch
            landmarks: Tensor
                Tensor containing the means of the landmarks
        """
        dists = euclidean_dist(latent, landmarks)
        min_dist, y_hat = torch.min(dists, 1)
        args_uniq = torch.unique(y_hat, sorted=True)
        args_count = torch.stack([(y_hat == x_u).sum() for x_u in args_uniq])

        if self.loss_metric == "dist":
            # Basic euclidean distance loss
            loss_val = torch.stack(
                [min_dist[y_hat == idx_class].mean(0) for idx_class in args_uniq]
            ).mean()

        elif self.loss_metric == "hyperbolic":
            # Transform Landmarks to hyperbolic ideal points
            h_landmarks = F.normalize(landmarks, p=2, dim=1)

            # Transform latent to hyperbolic space
            transformation_m = (
                (
                    torch.tanh(torch.norm(latent, p=2, dim=1) / 2)
                    / torch.norm(latent, p=2, dim=1)
                )
                .unsqueeze(dim=1)
                .expand(-1, latent.size(1))
            )
            h_latent = transformation_m * latent

            # Get tensor of closest landmarks in Euclidean space
            corr_land = h_landmarks[y_hat, :]

            # Buseman loss
            b_loss = torch.log(
                self.log1p_scale
                + torch.norm(corr_land - h_latent, p=2, dim=1) ** 2
                / (1 - torch.norm(h_latent, p=2, dim=1) ** 2)
            )

            # Overconfidence penalty loss
            overconf_loss = torch.log(1 - torch.norm(h_latent, p=2, dim=1) ** 2)

            # Calculate overall loss by taking mean of each cell
            loss_val = (b_loss - self.overconfidence_scale * overconf_loss).mean()

        elif self.loss_metric == "overlap":
            # Own idea of cell balls with center at landmark and radius of 95%-quantile
            assert False, "This loss may not work at current state"
            quantiles_view = self.landmarks_unlabeled_q.unsqueeze(0).expand(
                dists.size(0), dists.size(1)
            )
            overlap = torch.nan_to_num(
                torch.max(
                    torch.zeros_like(dists), (quantiles_view - dists) / quantiles_view
                )
            )
            id = torch.eye(len(landmarks), device=self.device)
            cross_entropy_dist = euclidean_dist(overlap, id)
            loss_val = cross_entropy_dist.min(1)[0].mean(0)

        elif self.loss_metric == "seurat":
            # Idea of using seurat distances for loss
            # See https://www.cell.com/cell/pdf/S0092-8674(19)30559-8.pdf
            assert False, "This loss may not work at current state"
            dists_t = 1 - (dists.T / dists.max(1)[0]).T
            prob = 1 - torch.exp(-dists_t / 4)
            prob = (prob.T / prob.sum(1)).T
            id = torch.eye(len(landmarks), device=self.device)
            cross_entropy_dist = euclidean_dist(prob, id)
            loss_val = cross_entropy_dist.min(1)[0].mean(0)

        elif self.loss_metric == "t":
            # Idea of using t-distribution as distance metric
            # See https://www.nature.com/articles/s41467-020-15851-3.pdf as example
            assert False, "This loss may not work at current state"
            q = t_dist(latent, landmarks, alpha=1)
            y_hat = q.argmax(1)
            args_uniq = torch.unique(y_hat, sorted=True)
            args_count = torch.stack([(y_hat == x_u).sum() for x_u in args_uniq])
            p = target_distribution(q)
            loss_val = kl_loss(q, p)

        else:
            assert False, (
                f"'{self.loss_metric}' is not a available as a loss function please choose "
                f"between 'dist','t' or 'seurat'!"
            )

        return loss_val, args_count
