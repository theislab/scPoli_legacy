import torch
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
from typing import Optional
import numpy as np

from scarches.models.trvae.trvae import trVAE
from scarches.models.trvae.losses import mse, mmd, zinb, nb
from scarches.models.trvae._utils import one_hot_encoder

from ._utils import euclidean_dist, get_overlap


class tranVAE(trVAE):
    def __init__(self,
                 input_dim: int,
                 conditions: list,
                 cell_types: list,
                 landmarks_labeled: Optional[dict] = None,
                 landmarks_unlabeled: Optional[dict] = None,
                 **trvae_kwargs,
                 ):
        super().__init__(
            input_dim,
            conditions,
            **trvae_kwargs)

        self.n_cell_types = len(cell_types)
        self.cell_types = cell_types
        self.cell_type_encoder = {k: v for k, v in zip(cell_types, range(len(cell_types)))}
        self.landmarks_labeled = {"mean": None, "var": None} if landmarks_labeled is None else landmarks_labeled
        self.landmarks_unlabeled = {"mean": None, "var": None} if landmarks_unlabeled is None else landmarks_unlabeled
        self.new_landmarks = None

        if self.landmarks_labeled["mean"] is not None:
            # Save indices of possible new landmarks to train
            self.new_landmarks = []
            for idx in range(self.n_cell_types - len(self.landmarks_labeled["mean"])):
                self.new_landmarks.append(len(self.landmarks_labeled["mean"]) + idx)

    def get_prob_matrix(self):
        # Returns (N-unlabeled-Landmarks x N-cell-types)-matrix with probabilities
        results = []
        for idx in range(len(self.landmarks_unlabeled["mean"])):
            unlabeled_result = []
            unlabeled_landmark_interval = torch.stack(
                (self.landmarks_unlabeled["mean"][idx,:] - self.landmarks_unlabeled["var"][idx,:],
                 self.landmarks_unlabeled["mean"][idx,:] + self.landmarks_unlabeled["var"][idx,:])
            )
            for cell_type in range(len(self.landmarks_labeled["mean"])):
                labeled_landmark_interval = torch.stack(
                    (self.landmarks_labeled["mean"][cell_type,:] - self.landmarks_labeled["var"][cell_type,:],
                     self.landmarks_labeled["mean"][cell_type,:] + self.landmarks_labeled["var"][cell_type,:])
                )
                prob = get_overlap(unlabeled_landmark_interval, labeled_landmark_interval)
                unlabeled_result.append(prob)
            results.append(unlabeled_result)
        return torch.tensor(results, device=self.landmarks_unlabeled["mean"].device)

    def classify(self, x, c=None, landmark=False):
        if landmark:
            latent = x
        else:
            latent = self.get_latent(x,c)

        distances = euclidean_dist(latent, self.landmarks_labeled["mean"])
        weighted_distances = F.softmax(-distances, dim=1)
        probs, preds = torch.max(weighted_distances, dim=1)

        return preds, probs

    def check_for_unseen(self):
        results = self.get_prob_matrix()
        probs, preds = torch.max(results, dim=1)
        return preds, probs

    def forward(self, x=None, batch=None, sizefactor=None, celltype=None, labeled=None):
        x_log = torch.log(1 + x)
        z1_mean, z1_log_var = self.encoder(x_log, batch)
        z1 = self.sampling(z1_mean, z1_log_var)
        outputs = self.decoder(z1, batch)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1).mean()
        elif self.recon_loss == "zinb":
            dec_mean_gamma, dec_dropout, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = -zinb(x=x, mu=dec_mean, theta=dispersion, pi=dec_dropout).sum(dim=-1).mean()
        elif self.recon_loss == "nb":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = -nb(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()

        z1_var = torch.exp(z1_log_var) + 1e-4
        kl_div = kl_divergence(
            Normal(z1_mean, torch.sqrt(z1_var)),
            Normal(torch.zeros_like(z1_mean), torch.ones_like(z1_var))
        ).sum(dim=1).mean()

        mmd_loss = torch.tensor(0.0, device=z1.device)
        if self.use_mmd:
            mmd_calculator = mmd(self.n_conditions, self.beta, self.mmd_boundary)
            if self.mmd_on == "z":
                mmd_loss = mmd_calculator(z1, batch)
            else:
                mmd_loss = mmd_calculator(y1, batch)

        return z1, recon_loss, kl_div, mmd_loss