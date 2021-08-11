import torch
from torch.distributions import Normal, kl_divergence, MultivariateNormal
import torch.nn.functional as F
from typing import Optional
import numpy as np

from scarches.models.trvae.trvae import trVAE
from scarches.models.trvae.losses import mse, mmd, zinb, nb
from scarches.models.trvae._utils import one_hot_encoder

from .._utils import euclidean_dist


class tranVAE(trVAE):
    def __init__(self,
                 input_dim: int,
                 conditions: list,
                 cell_types: list,
                 unknown_ct_names: Optional[list] = None,
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
        self.unknown_ct_names = unknown_ct_names
        if self.unknown_ct_names is not None:
            for unknown_ct in self.unknown_ct_names:
                self.cell_type_encoder[unknown_ct] = -1
        self.landmarks_labeled = {"mean": None, "q": None} if landmarks_labeled is None else landmarks_labeled
        self.landmarks_unlabeled = {"mean": None} if landmarks_unlabeled is None else landmarks_unlabeled
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

        #TODO: CALCULATE COV WITH CLUSTER CORRESPONDING CELLS INSTEAD OF SETTING TO ZERO
        new_landmark_q = torch.zeros(
            1, self.latent_dim, self.latent_dim,
            device=self.landmarks_labeled["q"].device, requires_grad=False
        )

        self.landmarks_labeled["mean"] = torch.cat(
            (self.landmarks_labeled["mean"], new_landmark),
            dim=0
        )
        self.landmarks_labeled["q"] = torch.cat(
            (self.landmarks_labeled["q"], new_landmark_q),
            dim=0
        )

    def classify(self, x, c=None, landmark=False, classes_list=None, metric="dist"):
        if landmark:
            latent = x
        else:
            latent = self.get_latent(x,c)

        dists = euclidean_dist(latent, self.landmarks_labeled["mean"][classes_list, :])

        if metric == "dist":
            # Idea of using euclidean distances for classification
            weighted_distances = F.softmax(-dists, dim=1)
            probs, preds = torch.max(weighted_distances, dim=1)
            preds = classes_list[preds]

        elif metric == "hyperbolic":
            # Transform Landmarks to hyperbolic ideal points
            h_landmarks = F.normalize(self.landmarks_labeled["mean"][classes_list, :], p=2, dim=1)

            # Transform latent to hyperbolic space
            transformation_m = (
                    torch.tanh(torch.norm(latent, p=2, dim=1) / 2) / torch.norm(latent, p=2, dim=1)
            ).unsqueeze(dim=1).expand(-1, latent.size(1))
            h_latent = transformation_m * latent

            # Get classification matrix n_cells x n_cell_types and get the predictions by max
            class_m = torch.matmul(
                h_latent / torch.norm(h_latent, p=2, dim=1).unsqueeze(dim=1).expand(-1, latent.size(1)),
                h_landmarks.T
            )
            class_m = F.normalize(class_m, p=1, dim=1)
            probs, preds = torch.max(class_m, dim=1)

        elif metric == "gaussian":
            probs = []
            for ct_class in classes_list:
                mean = self.landmarks_labeled["mean"][ct_class, :]
                cov_matrix = self.landmarks_labeled["q"][ct_class, :]
                # ID addition for stability
                # This has to be fixed in a better way maybe
                cov_matrix = cov_matrix + torch.eye(self.latent_dim, device=cov_matrix.device) * 1e-3
                #if torch.linalg.det(cov_matrix) == 0:
                #    cov_matrix = cov_matrix + torch.eye(self.latent_dim, device=cov_matrix.device) * 1e-3
                ct_distr = MultivariateNormal(mean, cov_matrix)
                probs.append(ct_distr.log_prob(latent).exp())

            probs = torch.stack(probs)
            probs = (probs / probs.sum(0)).T
            probs, preds = torch.max(probs, dim=1)
            preds = classes_list[preds]

        elif metric == "overlap":
            # Own idea of cell balls with center at landmark and radius of 95%-quantile
            assert False, "NEEDS CHECK"
            quantiles_view = self.landmarks_labeled["q"].unsqueeze(0).expand(dists.size(0), dists.size(1))
            #overlap = torch.max(torch.zeros_like(dists), (quantiles_view - dists))
            #overlap = 1 - (quantiles_view - overlap / quantiles_view)
            overlap = dists / quantiles_view
            overlap = (overlap.T / overlap.max(1)[0]).T
            overlap = 1 - overlap
            overlap = (overlap.T / overlap.sum(1)).T
            probs, preds = torch.max(overlap, dim=1)
            preds = classes_list[preds]

        elif metric == "seurat":
            # Idea of using seurat distances for classification
            # See https://www.cell.com/cell/pdf/S0092-8674(19)30559-8.pdf
            assert False, "NEEDS CHECK"
            dists_t = 1 - (dists.T / dists.max(1)[0]).T
            prob = 1 - torch.exp(-dists_t / 4)
            prob = (prob.T / prob.sum(1)).T
            probs, preds = torch.max(prob, dim=1)
            preds = classes_list[preds]

        else:
            assert False, f"'{metric}' is not a available as a loss function please choose " \
                          f"between 'exp', 'var' or 'seurat'!"

        return preds, probs

    def forward(self, x=None, batch=None, sizefactor=None, celltypes=None, labeled=None):
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