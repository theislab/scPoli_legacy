import torch
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
from typing import Optional
import numpy as np

from scarches.models.trvae.trvae import trVAE
from scarches.models.trvae.losses import mse, mmd, zinb, nb
from scarches.models.trvae._utils import one_hot_encoder

from ._utils import euclidean_dist


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
        self.landmarks_labeled = {"mean": None, "q": None} if landmarks_labeled is None else landmarks_labeled
        self.landmarks_unlabeled = {"mean": None, "q": None} if landmarks_unlabeled is None else landmarks_unlabeled
        self.new_landmarks = None

        if self.landmarks_labeled["mean"] is not None:
            # Save indices of possible new landmarks to train
            self.new_landmarks = []
            for idx in range(self.n_cell_types - len(self.landmarks_labeled["mean"])):
                self.new_landmarks.append(len(self.landmarks_labeled["mean"]) + idx)

    def add_new_cell_type(self, cell_type_name, landmarks):
        self.cell_types.append(cell_type_name)
        self.n_cell_types += 1
        self.cell_type_encoder = {k: v for k, v in zip(self.cell_types, range(len(self.cell_types)))}
        new_landmark = self.landmarks_unlabeled["mean"][landmarks].mean(0).unsqueeze(0)
        new_landmark_q = torch.tensor(
            1.0,
            device=self.landmarks_unlabeled["q"].device, requires_grad=False
        ).unsqueeze(0)
        self.landmarks_labeled["mean"] = torch.cat(
            (self.landmarks_labeled["mean"], new_landmark),
            dim=0
        )
        self.landmarks_labeled["q"] = torch.cat(
            (self.landmarks_labeled["q"], new_landmark_q),
            dim=0
        )

    def classify(self, x, c=None, landmark=False, metric="dist"):
        if landmark:
            latent = x
        else:
            latent = self.get_latent(x,c)

        dists = euclidean_dist(latent, self.landmarks_labeled["mean"])

        if metric == "dist":
            weighted_distances = F.softmax(-dists, dim=1)
            probs, preds = torch.max(weighted_distances, dim=1)
        elif metric == "seurat":
            dists_t = 1 - (dists.T / dists.max(1)[0]).T
            prob = 1 - torch.exp(-dists_t / 4)
            prob = (prob.T / prob.sum(1)).T
            probs, preds = torch.max(prob, dim=1)
        elif metric == "overlap":
            quantiles_view = self.landmarks_labeled["q"].unsqueeze(0).expand(dists.size(0), dists.size(1))

            #overlap = torch.max(torch.zeros_like(dists), (quantiles_view - dists))
            #overlap = 1 - (quantiles_view - overlap / quantiles_view)

            overlap = dists / quantiles_view
            overlap = (overlap.T / overlap.max(1)[0]).T
            overlap = 1 - overlap

            overlap = (overlap.T / overlap.sum(1)).T
            probs, preds = torch.max(overlap, dim=1)
        else:
            assert False, f"'{metric}' is not a available as a loss function please choose " \
                          f"between 'exp', 'var' or 'seurat'!"

        return preds, probs

    def check_for_unseen(self):
        landmark_dists = euclidean_dist(self.landmarks_unlabeled["mean"], self.landmarks_labeled["mean"])
        quantile_sum = self.landmarks_unlabeled["q"].unsqueeze(1).expand(landmark_dists.size(0),
                                                                         landmark_dists.size(1)) + \
                       self.landmarks_labeled["q"].unsqueeze(0).expand(landmark_dists.size(0),
                                                                       landmark_dists.size(1))

        #overlap = torch.max(torch.zeros_like(landmark_dists), (quantile_sum - landmark_dists)/landmark_dists)
        #overlap = torch.nan_to_num((overlap.T / overlap.sum(1))).T

        overlap = landmark_dists / quantile_sum
        overlap = (overlap.T / overlap.max(1)[0]).T
        overlap = 1 - overlap
        overlap = (overlap.T / overlap.sum(1)).T
        probs, preds = torch.max(overlap, dim=1)
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