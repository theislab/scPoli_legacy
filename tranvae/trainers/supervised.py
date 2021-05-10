import torch
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from torch.distributions import Normal
from scarches.trainers.trvae.trainer import Trainer
from scarches.trainers.trvae._utils import make_dataset

from ._utils import euclidean_dist, t_dist, target_distribution, kl_loss


class tranVAETrainer(Trainer):
    """ScArches Unsupervised Trainer class. This class contains the implementation of the unsupervised CVAE/TRVAE
       Trainer.

           Parameters
           ----------
           model: trVAE
                Number of input features (i.e. gene in case of scRNA-seq).
           adata: : `~anndata.AnnData`
                Annotated data matrix.
           condition_key: String
                column name of conditions in `adata.obs` data frame.           cell_type_key: String
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
            n_clusters: int = None,
            clustering: str = 'kmeans',
            use_unlabeled_loss: bool = True,
            eta: float = 1000,
            eta_epoch_anneal: int = None,
            tau: float = 0,
            labeled_indices: list = None,
            **kwargs
    ):

        super().__init__(model, adata, **kwargs)
        self.loss_metric = "mars"
        self.landmarks_labeled = None
        self.eta = eta
        self.eta_epoch_anneal = eta_epoch_anneal
        self.tau = tau
        self.clustering = clustering
        self.n_clusters = n_clusters
        self.use_unlabeled_loss = use_unlabeled_loss
        self.n_labeled = self.model.n_cell_types
        self.lndmk_optim = None

        # Set indices for labeled data
        if labeled_indices is None:
            self.labeled_indices = range(len(adata))
        else:
            self.labeled_indices = labeled_indices
        self.update_labeled_indices(self.labeled_indices)

        # Parse landmarks from model into right format
        if self.model.landmarks_labeled is not None:
            landmarks_means = []
            for landmark in self.model.landmarks_labeled:
                landmarks_means.append(landmark.mean)
            self.landmarks_labeled = torch.stack(landmarks_means)
        if self.landmarks_labeled is not None:
            self.landmarks_labeled = self.landmarks_labeled.to(device=self.device)
        self.landmarks_unlabeled = self.model.landmarks_unlabeled
        if self.landmarks_unlabeled is not None:
            self.landmarks_unlabeled = torch.tensor(self.landmarks_unlabeled, device=self.device)

    def calc_eta_coeff(self):
        """Calculates current alpha coefficient for alpha annealing.

           Parameters
           ----------

           Returns
           -------
           Current annealed alpha value
        """
        if self.alpha_epoch_anneal is not None:
            eta_coeff = min(self.epoch / self.eta_epoch_anneal, 1)
        else:
            eta_coeff = 1
        return eta_coeff

    def update_labeled_indices(self, labeled_indices):
        self.labeled_indices = labeled_indices
        self.train_data, self.valid_data = make_dataset(
            self.adata,
            train_frac=self.train_frac,
            use_stratified_split=self.use_stratified_split,
            condition_key=self.condition_key,
            cell_type_key=self.cell_type_key,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
            labeled_indices=self.labeled_indices,
        )

    def before_loop(self, lr, eps):
        self.initialize_landmarks()
        if 0 in self.train_data.labeled_vector.unique().tolist():
            self.lndmk_optim = torch.optim.Adam(
                params=self.landmarks_unlabeled,
                lr=lr,
                eps=eps,
                weight_decay=self.weight_decay
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
            latents += [latent]
        latent = torch.cat(latents)
        return latent

    def loss(self, total_batch=None):
        latent, recon_loss, kl_loss, mmd_loss = self.model(**total_batch)

        # Calculate classifier loss for labeled/unlabeled data
        label_categories = total_batch["labeled"].unique().tolist()
        landmark_loss = torch.tensor(0, device=self.device, requires_grad=False)
        unlabeled_loss = torch.tensor(0, device=self.device, requires_grad=False)
        labeled_loss = torch.tensor(0, device=self.device, requires_grad=False)

        # Calculate landmark loss for unlabeled data
        if 0 in label_categories and self.use_unlabeled_loss:
            unlabeled_loss, _ = self.landmark_unlabeled_loss(
                latent[total_batch['labeled'] == 0],
                torch.stack(self.landmarks_unlabeled).squeeze(),
                self.tau,
            )
            landmark_loss = landmark_loss + unlabeled_loss

        # Calculate landmark loss for labeled data
        if 1 in label_categories:
            labeled_loss, labeled_accuracy = self.landmark_labeled_loss(
                latent[total_batch['labeled'] == 1],
                self.landmarks_labeled,
                total_batch["celltype"][total_batch['labeled'] == 1],
            )
            landmark_loss = landmark_loss + labeled_loss

        # Loss addition and Logs
        classifier_loss = self.calc_eta_coeff() * self.eta * landmark_loss
        trvae_loss = recon_loss + self.calc_alpha_coeff() * kl_loss + mmd_loss
        loss = trvae_loss + classifier_loss
        self.iter_logs["loss"].append(loss.item())
        self.iter_logs["unweighted_loss"].append(
            recon_loss.item() + kl_loss.item() + mmd_loss.item() + landmark_loss.item()
        )
        self.iter_logs["trvae_loss"].append(trvae_loss.item())
        self.iter_logs["classifier_loss"].append(classifier_loss.item())
        if 0 in label_categories and self.use_unlabeled_loss:
            self.iter_logs["unlabeled_loss"].append(unlabeled_loss.item())
        if 1 in label_categories:
            self.iter_logs["labeled_loss"].append(labeled_loss.item())
            self.iter_logs["accuracy"].append(labeled_accuracy.item())
        return loss

    def on_epoch_end(self):
        self.model.eval()

        latent = self.get_latent_train()
        label_categories = self.train_data.labeled_vector.unique().tolist()

        # Update unlabeled landmark positions
        if 0 in label_categories:
            for landmk in self.landmarks_unlabeled:
                landmk.requires_grad = True
            self.lndmk_optim.zero_grad()
            update_loss, args_count = self.landmark_unlabeled_loss(
                latent[self.train_data.labeled_vector == 0],
                torch.stack(self.landmarks_unlabeled).squeeze(),
                self.tau,
            )
            update_loss.backward()
            self.lndmk_optim.step()
            for landmk in self.landmarks_unlabeled:
                landmk.requires_grad = False

        # Update labeled landmark positions
        if 1 in label_categories:
            self.landmarks_labeled = self.update_labeled_landmarks(
                latent[self.train_data.labeled_vector == 1],
                self.train_data.cell_types[self.train_data.labeled_vector == 1],
                self.landmarks_labeled,
                self.tau,
                self.model.new_landmarks
            )
        self.model.train()

        super().on_epoch_end()

    def after_loop(self):
        latent = self.get_latent_train()
        landmarks_distr, landmarks_normalize = self.update_landmark_distr(
            latent[self.train_data.labeled_vector == 1],
            self.train_data.cell_types[self.train_data.labeled_vector == 1],
            self.model.new_landmarks
        )
        self.model.landmarks_labeled = landmarks_distr
        self.model.landmarks_normalize = landmarks_normalize

        if 0 in self.train_data.labeled_vector.unique().tolist():
            self.model.landmarks_unlabeled = torch.stack(self.landmarks_unlabeled).squeeze()
        else:
            self.model.landmarks_unlabeled = self.landmarks_unlabeled

    def initialize_landmarks(self):
        # Compute Latent of whole train data
        latent = self.get_latent_train()

        # Init labeled Landmarks if labeled data existent
        if 1 in self.train_data.labeled_vector.unique().tolist():
            labeled_latent = latent[self.train_data.labeled_vector == 1]
            labeled_cell_types = self.train_data.cell_types[self.train_data.labeled_vector == 1]
            if self.landmarks_labeled is not None:
                with torch.no_grad():
                    if len(self.model.new_landmarks) > 0:
                        for value in self.model.new_landmarks:
                            indices = labeled_cell_types.eq(value).nonzero()
                            landmark = labeled_latent[indices].mean(0)
                            self.landmarks_labeled = torch.cat([self.landmarks_labeled, landmark])
            else:
                self.landmarks_labeled = self.update_labeled_landmarks(
                    latent[self.train_data.labeled_vector == 1],
                    self.train_data.cell_types[self.train_data.labeled_vector == 1],
                    None,
                    self.tau
                )

        # Init unlabeled Landmarks if unlabeled data existent
        if 0 in self.train_data.labeled_vector.unique().tolist():
            # Full latent here (also labeled data), maybe change back to only unlabeled
            # lat_array = latent[self.train_data.labeled_vector == 0].cpu().detach().numpy()
            lat_array = latent.cpu().detach().numpy()

            if self.clustering == 'kmeans' and self.n_clusters is not None:
                print(f'Initializing unlabeled landmarks with KMeans-Clustering with a given number of'
                      f'{self.n_clusters} clusters.')
                k_means = KMeans(n_clusters=self.n_clusters).fit(lat_array)
                k_means_lndmk = torch.tensor(k_means.cluster_centers_, device=self.device)

                self.landmarks_unlabeled = [
                    torch.zeros(
                        size=(1, self.model.latent_dim),
                        requires_grad=True,
                        device=self.device)
                    for _ in range(self.n_clusters)
                ]

                with torch.no_grad():
                    [self.landmarks_unlabeled[i].copy_(k_means_lndmk[i, :]) for i in range(k_means_lndmk.shape[0])]
            else:
                if self.clustering == 'kmeans' and self.n_clusters is None:
                    print(f'Initializing unlabeled landmarks with Louvain-Clustering because no value for the'
                          f'number of clusters was given.')
                else:
                    print(f'Initializing unlabeled landmarks with Louvain-Clustering with an unknown number of '
                          f'clusters.')
                lat_adata = sc.AnnData(lat_array)
                sc.pp.neighbors(lat_adata)
                sc.tl.louvain(lat_adata)

                # Taken from DESC model
                features = pd.DataFrame(lat_adata.X, index=np.arange(0, lat_adata.shape[0]))
                Group = pd.Series(
                    np.asarray(lat_adata.obs['louvain'],dtype=int),
                    index=np.arange(0, lat_adata.shape[0]),
                    name="Group"
                )
                Mergefeature = pd.concat([features, Group], axis=1)
                cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

                self.n_clusters = cluster_centers.shape[0]
                print(f'Louvain Clustering succesful. Found {self.n_clusters} clusters.')
                louvain_lndmk = torch.tensor(cluster_centers, device=self.device)

                self.landmarks_unlabeled = [
                    torch.zeros(
                        size=(1, self.model.latent_dim),
                        requires_grad=True,
                        device=self.device)
                    for _ in range(self.n_clusters)
                ]

                with torch.no_grad():
                    [self.landmarks_unlabeled[i].copy_(louvain_lndmk[i, :]) for i in range(louvain_lndmk.shape[0])]

    def update_labeled_landmarks(self, latent, labels, previous_landmarks, tau, mask=None):
        with torch.no_grad():
            unique_labels = torch.unique(labels, sorted=True)
            landmarks_mean = None
            for value in range(self.n_labeled):
                if (mask is None or value in mask) and value in unique_labels:
                    indices = labels.eq(value).nonzero(as_tuple=False)
                    landmark = latent[indices].mean(0)
                    landmarks_mean = torch.cat([landmarks_mean, landmark]) if landmarks_mean is not None else landmark
                else:
                    landmark = previous_landmarks[value].unsqueeze(0)
                    landmarks_mean = torch.cat([landmarks_mean, landmark]) if landmarks_mean is not None else landmark

            if previous_landmarks is None or tau == 0:
                return landmarks_mean

            previous_landmarks_sum = previous_landmarks.sum(0)
            n_landmarks = previous_landmarks.shape[0]
            landmarks_distance_partial = (tau / (n_landmarks - 1)) * torch.stack(
                [previous_landmarks_sum - landmark for landmark in previous_landmarks])
            landmarks = (1 / (1 - tau)) * (landmarks_mean - landmarks_distance_partial)

        return landmarks

    def landmark_labeled_loss(self, latent, landmarks, labels):
        n_samples = latent.shape[0]
        unique_labels = torch.unique(labels, sorted=True)
        distances = euclidean_dist(latent, landmarks)
        loss = None
        for value in unique_labels:
            indices = labels.eq(value).nonzero(as_tuple=False)
            label_loss = distances[indices, value].sum(0)
            loss = torch.cat([loss, label_loss]) if loss is not None else label_loss
        loss = loss.sum() / n_samples

        _, y_pred = torch.max(-distances, dim=1)

        accuracy = y_pred.eq(labels.squeeze()).float().mean()

        return loss, accuracy

    def unlabeled_loss_basic(self, latent, landmarks):
        if self.loss_metric == "mars":
            dists = euclidean_dist(latent, landmarks)
            min_dist = torch.min(dists, 1)

            y_hat = min_dist[1]
            args_uniq = torch.unique(y_hat, sorted=True)
            args_count = torch.stack([(y_hat == x_u).sum() for x_u in args_uniq])

            min_dist = min_dist[0]  # get_distances

            loss_val = torch.stack([min_dist[y_hat == idx_class].mean(0) for idx_class in args_uniq]).mean()
        else:
            q = t_dist(latent, landmarks, alpha=1)
            p = target_distribution(q)
            loss_val = kl_loss(q, p)
            y_pred = q.argmax(1)
            args_uniq = torch.unique(y_pred, sorted=True)
            args_count = torch.stack([(y_pred == x_u).sum() for x_u in args_uniq])
        return loss_val, args_count

    def landmark_unlabeled_loss(self, latent, landmarks, tau):
        loss_val_test, args_count = self.unlabeled_loss_basic(latent, landmarks)
        if tau > 0:
            dists = euclidean_dist(landmarks, landmarks)
            nproto = landmarks.shape[0]
            loss_val2 = - torch.sum(dists) / (nproto * nproto - nproto)

            loss_val_test += tau * loss_val2

        return loss_val_test, args_count

    def update_landmark_distr(self, latent, labels,  mask=None):
        landmarks = []
        landmarks_normalize = []
        for idx,landmark in enumerate(self.landmarks_labeled):
            if (mask is None or idx in mask) and 1 in self.train_data.labeled_vector.unique().tolist():
                ct_latent = latent[labels == idx]
                ct_std = torch.std(ct_latent, dim=0)
                ct_distr = Normal(landmark.cpu(), ct_std.cpu())
                landmarks.append(ct_distr)
                landmarks_normalize.append(torch.mean(ct_distr.log_prob(landmark.cpu()).exp(), axis=0))
            else:
                ct_distr = self.model.landmarks_labeled[idx]
                landmarks.append(ct_distr)
                landmarks_normalize.append(torch.mean(ct_distr.log_prob(landmark.cpu()).exp(), axis=0))
        landmarks_normalize = torch.stack(landmarks_normalize)

        return landmarks, landmarks_normalize
