import torch
from torch.nn import NLLLoss
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scarches.trainers.trvae.trainer import Trainer
from scarches.trainers.trvae._utils import make_dataset

from ._utils import euclidean_dist, t_dist, target_distribution, kl_loss, get_overlap


class tranVAETrainer(Trainer):
    """
    ScArches Unsupervised Trainer class. This class contains the implementation of the unsupervised CVAE/TRVAE
    Trainer.

    Parameters
    ----------
    model: trVAE
        Number of input features (i.e. gene in case of scRNA-seq).
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
    use_stratified_split: Boolean
        If `True`, the train and validation data will be constructed in such a way that both have same distribution
        of conditions in the data.
    monitor: Boolean
        If `True', the progress of the training will be printed after each epoch.
    n_workers: Integer
        Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
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
            labeled_loss_metric: str = "dist",
            unlabeled_loss_metric: str = "dist",
            unlabeled_weight: float = 0.001,
            eta: float = 1,
            tau: float = 0,
            **kwargs
    ):

        super().__init__(model, adata, **kwargs)
        self.labeled_loss_metric = labeled_loss_metric
        self.unlabeled_loss_metric = unlabeled_loss_metric
        self.eta = eta
        self.tau = tau
        self.clustering = clustering
        self.n_clusters = n_clusters
        self.unlabeled_weight = unlabeled_weight
        self.clustering_res = clustering_res
        self.pretraining_epochs = pretraining_epochs
        self.use_early_stopping_orig = self.use_early_stopping
        self.quantile = 0.95
        self.cross_entropy = NLLLoss()

        self.landmarks_labeled = None
        self.landmarks_labeled_q = None
        self.landmarks_unlabeled = None
        self.landmarks_unlabeled_q = None
        self.best_landmarks_labeled = None
        self.best_landmarks_labeled_q = None
        self.best_landmarks_unlabeled = None
        self.best_landmarks_unlabeled_q = None
        self.lndmk_optim = None

        # Set indices for labeled data
        if labeled_indices is None:
            self.labeled_indices = range(len(adata))
        else:
            self.labeled_indices = labeled_indices
        self.update_labeled_indices(self.labeled_indices)

        # Parse landmarks from model into right format
        if self.model.landmarks_labeled["mean"] is not None:
            self.landmarks_labeled = self.model.landmarks_labeled["mean"]
            self.landmarks_labeled_q = self.model.landmarks_labeled["q"]
        if self.landmarks_labeled is not None:
            self.landmarks_labeled = self.landmarks_labeled.to(device=self.device)
            self.landmarks_labeled_q = self.landmarks_labeled_q.to(device=self.device)

    def update_labeled_indices(self, labeled_indices):
        self.labeled_indices = labeled_indices
        self.train_data, self.valid_data = make_dataset(
            self.adata,
            train_frac=self.train_frac,
            use_stratified_split=self.use_stratified_split,
            condition_key=self.condition_key,
            cell_type_keys=self.cell_type_keys,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
            labeled_indices=self.labeled_indices,
        )

    def get_latent_train(self):
        latents = []
        indices = torch.arange(self.train_data.data.size(0), device=self.device)
        subsampled_indices = indices.split(self.batch_size)
        for batch in subsampled_indices:
            latent = self.model.get_latent(
                self.train_data.data[batch, :].to(self.device),
                self.train_data.conditions[batch].to(self.device)
            )
            latents += [latent.cpu().detach()]
        latent = torch.cat(latents)
        return latent.to(self.device)

    def initialize_landmarks(self):
        # Compute Latent of whole train data
        latent = self.get_latent_train()

        # Init labeled Landmarks if labeled data existent
        if 1 in self.train_data.labeled_vector.unique().tolist():
            labeled_latent = latent[self.train_data.labeled_vector == 1]
            labeled_cell_types = self.train_data.cell_types[self.train_data.labeled_vector == 1, :]
            if self.landmarks_labeled is not None:
                with torch.no_grad():
                    if len(self.model.new_landmarks) > 0:
                        for value in self.model.new_landmarks:
                            indices = labeled_cell_types.eq(value).nonzero(as_tuple=False)[:, 0]
                            landmark = labeled_latent[indices].mean(0)
                            dist = euclidean_dist(labeled_latent[indices], landmark)
                            landmark_q = torch.quantile(dist, self.quantile).unsqueeze(0)
                            self.landmarks_labeled = torch.cat([self.landmarks_labeled, landmark])
                            self.landmarks_labeled_q = torch.cat([self.landmarks_labeled_q, landmark_q])
            else:
                self.landmarks_labeled, self.landmarks_labeled_q = self.update_labeled_landmarks(
                    latent[self.train_data.labeled_vector == 1],
                    self.train_data.cell_types[self.train_data.labeled_vector == 1, :],
                    None,
                    None,
                )

        # Init unlabeled Landmarks if unlabeled data existent
        if 0 in self.train_data.labeled_vector.unique().tolist() or self.model.unknown_ct_names is not None:
            lat_array = latent.cpu().detach().numpy()

            if self.clustering == "kmeans" and self.n_clusters is not None:
                print(f"\nInitializing unlabeled landmarks with KMeans-Clustering with a given number of"
                      f"{self.n_clusters} clusters.")
                k_means = KMeans(n_clusters=self.n_clusters).fit(lat_array)
                k_means_lndmk = torch.tensor(k_means.cluster_centers_, device=self.device)

                self.landmarks_unlabeled = [
                    torch.zeros(
                        size=(1, self.model.latent_dim),
                        requires_grad=False,
                        device=self.device)
                    for _ in range(self.n_clusters)
                ]

                with torch.no_grad():
                    [self.landmarks_unlabeled[i].copy_(k_means_lndmk[i, :]) for i in range(k_means_lndmk.shape[0])]
            else:
                if self.clustering == "kmeans" and self.n_clusters is None:
                    print(f"\nInitializing unlabeled landmarks with Leiden-Clustering because no value for the"
                          f"number of clusters was given.")
                else:
                    print(f"\nInitializing unlabeled landmarks with Leiden-Clustering with an unknown number of "
                          f"clusters.")
                lat_adata = sc.AnnData(lat_array)
                sc.pp.neighbors(lat_adata)
                sc.tl.leiden(lat_adata, resolution=self.clustering_res)

                # Taken from DESC model
                features = pd.DataFrame(lat_adata.X, index=np.arange(0, lat_adata.shape[0]))
                Group = pd.Series(
                    np.asarray(lat_adata.obs["leiden"],dtype=int),
                    index=np.arange(0, lat_adata.shape[0]),
                    name="Group"
                )
                Mergefeature = pd.concat([features, Group], axis=1)
                cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

                self.n_clusters = cluster_centers.shape[0]
                print(f"Leiden Clustering succesful. Found {self.n_clusters} clusters.")
                leiden_lndmk = torch.tensor(cluster_centers, device=self.device)

                self.landmarks_unlabeled = [
                    torch.zeros(
                        size=(1, self.model.latent_dim),
                        requires_grad=False,
                        device=self.device)
                    for _ in range(self.n_clusters)
                ]

                with torch.no_grad():
                    [self.landmarks_unlabeled[i].copy_(leiden_lndmk[i, :]) for i in range(leiden_lndmk.shape[0])]
                    dists = euclidean_dist(latent, torch.stack(self.landmarks_unlabeled).squeeze())
                    min_dist, y_hat = torch.min(dists, 1)
                    quantiles = []
                    for idx_class in range(len(torch.stack(self.landmarks_unlabeled).squeeze())):
                        if idx_class in y_hat:
                            quantiles.append(torch.quantile(min_dist[y_hat == idx_class], self.quantile, dim=0).unsqueeze(0))
                        else:
                            quantiles.append(torch.tensor(0.0, device=self.device).unsqueeze(0))
                    self.landmarks_unlabeled_q = torch.stack(quantiles)

    def on_epoch_begin (self, lr, eps):
        if self.epoch == self.pretraining_epochs:
            self.initialize_landmarks()
            if 0 in self.train_data.labeled_vector.unique().tolist() or self.model.unknown_ct_names is not None:
                self.lndmk_optim = torch.optim.Adam(
                    params=self.landmarks_unlabeled,
                    lr=lr,
                    eps=eps,
                    weight_decay=self.weight_decay
                )
        if self.epoch < self.pretraining_epochs:
            self.use_early_stopping = False
        if self.use_early_stopping_orig and self.epoch >= self.pretraining_epochs:
            self.use_early_stopping = True
        if self.epoch >= self.pretraining_epochs and self.epoch - 1 == self.best_epoch:
            self.best_landmarks_labeled = self.landmarks_labeled
            self.best_landmarks_labeled_q = self.landmarks_labeled_q
            self.best_landmarks_unlabeled = self.landmarks_unlabeled
            self.best_landmarks_unlabeled_q = self.landmarks_unlabeled_q

    def loss(self, total_batch=None):
        latent, recon_loss, kl_loss, mmd_loss = self.model(**total_batch)

        # Calculate classifier loss for labeled/unlabeled data
        label_categories = total_batch["labeled"].unique().tolist()
        landmark_loss = torch.tensor(0.0, device=self.device)
        unlabeled_loss = torch.tensor(0.0, device=self.device)
        labeled_loss = torch.tensor(0.0, device=self.device)
        if self.epoch >= self.pretraining_epochs:
            # Calculate landmark loss for unlabeled data
            if self.landmarks_unlabeled is not None and self.unlabeled_weight > 0:
                unlabeled_loss, _ = self.landmark_unlabeled_loss(
                    latent,
                    torch.stack(self.landmarks_unlabeled).squeeze(),
                )
                landmark_loss = landmark_loss + self.unlabeled_weight * unlabeled_loss

            # Calculate landmark loss for labeled data
            if 1 in label_categories:
                labeled_loss = self.landmark_labeled_loss(
                    latent[total_batch["labeled"] == 1],
                    self.landmarks_labeled,
                    total_batch["celltypes"][total_batch["labeled"] == 1, :],
                )
                landmark_loss = landmark_loss + labeled_loss

        # Loss addition and Logs
        classifier_loss = self.eta * landmark_loss
        trvae_loss = recon_loss + self.calc_alpha_coeff() * kl_loss + mmd_loss
        loss = trvae_loss + classifier_loss
        self.iter_logs["loss"].append(loss.item())
        self.iter_logs["unweighted_loss"].append(
            recon_loss.item() + kl_loss.item() + mmd_loss.item() + landmark_loss.item()
        )
        self.iter_logs["trvae_loss"].append(trvae_loss.item())
        if self.epoch >= self.pretraining_epochs:
            self.iter_logs["classifier_loss"].append(classifier_loss.item())
            if 0 in label_categories or self.model.unknown_ct_names is not None:
                self.iter_logs["unlabeled_loss"].append(unlabeled_loss.item())
            if 1 in label_categories:
                self.iter_logs["labeled_loss"].append(labeled_loss.item())
        return loss

    def on_epoch_end(self):
        self.model.eval()

        if self.epoch >= self.pretraining_epochs:
            latent = self.get_latent_train()
            label_categories = self.train_data.labeled_vector.unique().tolist()

            # Update labeled landmark positions
            if 1 in label_categories:
                self.landmarks_labeled, self.landmarks_labeled_q = self.update_labeled_landmarks(
                    latent[self.train_data.labeled_vector == 1],
                    self.train_data.cell_types[self.train_data.labeled_vector == 1, :],
                    self.landmarks_labeled,
                    self.landmarks_labeled_q,
                    self.model.new_landmarks
                )

            # Update unlabeled landmark positions
            if 0 in label_categories or self.model.unknown_ct_names is not None:
                for landmk in self.landmarks_unlabeled:
                    landmk.requires_grad = True
                self.lndmk_optim.zero_grad()
                update_loss, args_count = self.landmark_unlabeled_loss(
                    latent,
                    torch.stack(self.landmarks_unlabeled).squeeze(),
                    update_q=True,
                    update_pos=True,
                )
                update_loss.backward()
                self.lndmk_optim.step()
                for landmk in self.landmarks_unlabeled:
                    landmk.requires_grad = False

        self.model.train()
        super().on_epoch_end()

    def after_loop(self):
        if self.best_state_dict is not None and self.reload_best:
            self.landmarks_labeled = self.best_landmarks_labeled
            self.landmarks_labeled_q = self.best_landmarks_labeled_q
            self.landmarks_unlabeled = self.best_landmarks_unlabeled
            self.landmarks_unlabeled_q = self.best_landmarks_unlabeled_q

        label_categories = self.train_data.labeled_vector.unique().tolist()
        if 0 in label_categories or self.model.unknown_ct_names is not None:
            latent = self.get_latent_train()
            landmarks = torch.stack(self.landmarks_unlabeled).squeeze()

            dists = euclidean_dist(latent, landmarks)
            min_dist, y_hat = torch.min(dists, 1)

            quantiles = []
            for idx_class in range(len(landmarks)):
                if idx_class in y_hat:
                    quantiles.append(torch.quantile(min_dist[y_hat == idx_class], self.quantile, dim=0).unsqueeze(0))
                else:
                    quantiles.append(torch.tensor(0.0, device=self.device).unsqueeze(0))
            self.landmarks_unlabeled_q = torch.stack(quantiles)

        self.model.landmarks_labeled["mean"] = self.landmarks_labeled
        self.model.landmarks_labeled["q"] = self.landmarks_labeled_q

        if 0 in self.train_data.labeled_vector.unique().tolist() or self.model.unknown_ct_names is not None:
            self.model.landmarks_unlabeled["mean"] = torch.stack(self.landmarks_unlabeled).squeeze()
            self.model.landmarks_unlabeled["q"] = self.landmarks_unlabeled_q
        else:
            self.model.landmarks_unlabeled["mean"] = self.landmarks_unlabeled
            self.model.landmarks_unlabeled["q"] = self.landmarks_unlabeled_q

    def update_labeled_landmarks(self, latent, labels, previous_landmarks, previous_landmarks_q, mask=None):
        with torch.no_grad():
            unique_labels = torch.unique(labels, sorted=True)
            landmarks_mean = None
            landmarks_q = None
            for value in range(self.model.n_cell_types):
                if (mask is None or value in mask) and value in unique_labels:
                    indices = labels.eq(value).nonzero(as_tuple=False)[:, 0]
                    landmark = latent[indices, :].mean(0).unsqueeze(0)
                    dist = euclidean_dist(latent[indices], landmark)
                    landmark_q = torch.quantile(dist, self.quantile, dim=0).unsqueeze(0)
                    landmarks_mean = torch.cat(
                        [landmarks_mean, landmark]) if landmarks_mean is not None else landmark
                    landmarks_q = torch.cat(
                        [landmarks_q, landmark_q]) if landmarks_q is not None else landmark_q
                else:
                    landmark = previous_landmarks[value].unsqueeze(0)
                    landmark_q = previous_landmarks_q[value].unsqueeze(0)
                    landmarks_mean = torch.cat(
                        [landmarks_mean, landmark]) if landmarks_mean is not None else landmark
                    landmarks_q = torch.cat(
                        [landmarks_q, landmark_q]) if landmarks_q is not None else landmark_q
        return landmarks_mean, landmarks_q

    def landmark_labeled_loss(self, latent, landmarks, labels):
        unique_labels = torch.unique(labels, sorted=True)
        distances = euclidean_dist(latent, landmarks)
        loss = torch.tensor(0.0, device=self.device)

        # Basic distance loss works with hierarchy
        if self.labeled_loss_metric == "dist":
            for value in unique_labels:
                if value == -1:
                    continue
                indices = labels.eq(value).nonzero(as_tuple=False)[:,0]
                label_loss = distances[indices, value].sum(0) / len(indices)
                loss += label_loss

        # Alternative to distance loss, however may not work with hierarchy
        elif self.labeled_loss_metric == "overlap":
            id = torch.eye(len(landmarks), device=self.device)
            truth_id = id[labels]
            quantiles_view = self.landmarks_labeled_q.unsqueeze(0).expand(distances.size(0), distances.size(1))
            overlap = torch.max(torch.zeros_like(distances), (quantiles_view - distances) / quantiles_view)
            loss = torch.pow(truth_id - overlap, 2).sum(1).mean(0)

        # Alternative to distance loss, however may not work with hierarchy
        elif self.labeled_loss_metric == "seurat":
            dists_t = 1 - (distances.T / distances.max(1)[0]).T
            prob = 1 - torch.exp(-dists_t / 4)
            prob = (prob.T / prob.sum(1)).T
            id = torch.eye(len(landmarks), device=self.device)
            truth_id = id[labels]
            loss = torch.pow(truth_id - prob, 2).sum(1).mean(0)

        else:
            assert False, f"'{self.labeled_loss_metric}' is not a available as a loss function please choose " \
                          f"between 'dist','t' or 'seurat'!"

        return loss

    def landmark_unlabeled_loss(self, latent, landmarks, update_q=False, update_pos=False):
        dists = euclidean_dist(latent, landmarks)
        min_dist, y_hat = torch.min(dists, 1)
        args_uniq = torch.unique(y_hat, sorted=True)
        args_count = torch.stack([(y_hat == x_u).sum() for x_u in args_uniq])

        with torch.no_grad():
            if update_q:
                quantiles = []
                for idx_class in range(len(landmarks)):
                    if idx_class in y_hat:
                        quantiles.append(torch.quantile(min_dist[y_hat == idx_class], self.quantile, dim=0).unsqueeze(0))
                    else:
                        quantiles.append(torch.tensor(0.0, device=self.device).unsqueeze(0))
                self.landmarks_unlabeled_q = torch.stack(quantiles)

        if self.unlabeled_loss_metric == "dist":
            loss_val = torch.stack([min_dist[y_hat == idx_class].mean(0) for idx_class in args_uniq]).mean()
        elif self.unlabeled_loss_metric == "t":
            q = t_dist(latent, landmarks, alpha=1)
            y_hat = q.argmax(1)
            args_uniq = torch.unique(y_hat, sorted=True)
            args_count = torch.stack([(y_hat == x_u).sum() for x_u in args_uniq])
            p = target_distribution(q)
            loss_val = kl_loss(q, p)
        elif self.unlabeled_loss_metric == "overlap":
            quantiles_view = self.landmarks_unlabeled_q.unsqueeze(0).expand(dists.size(0), dists.size(1))
            overlap = torch.nan_to_num(torch.max(torch.zeros_like(dists), (quantiles_view - dists) / quantiles_view))
            id = torch.eye(len(landmarks), device=self.device)
            cross_entropy_dist = euclidean_dist(overlap, id)
            loss_val = cross_entropy_dist.min(1)[0].mean(0)
        elif self.unlabeled_loss_metric == "seurat":
            dists_t = 1 - (dists.T / dists.max(1)[0]).T
            prob = 1 - torch.exp(-dists_t / 4)
            prob = (prob.T / prob.sum(1)).T
            id = torch.eye(len(landmarks), device=self.device)
            cross_entropy_dist = euclidean_dist(prob, id)
            loss_val = cross_entropy_dist.min(1)[0].mean(0)
        else:
            assert False, f"'{self.unlabeled_loss_metric}' is not a available as a loss function please choose " \
                          f"between 'dist','t' or 'seurat'!"

        # Check if unlabeled landmark is close to labeled landmark
        if update_pos and self.tau != 0:
            landmark_dists = euclidean_dist(landmarks, self.landmarks_labeled)
            quantile_sum = self.landmarks_unlabeled_q.unsqueeze(1).expand(landmark_dists.size(0),
                                                                             landmark_dists.size(1)) + \
                           self.landmarks_labeled_q.unsqueeze(0).expand(landmark_dists.size(0),
                                                                           landmark_dists.size(1))

            overlap = torch.max(torch.zeros_like(landmark_dists),
                                (quantile_sum - landmark_dists) / (0.5 * quantile_sum))
            distance_loss = (landmark_dists * overlap).sum(1).mean(0)
            loss_val += self.tau * distance_loss

        return loss_val, args_count
