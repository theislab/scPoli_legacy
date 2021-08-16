import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, kl_divergence, MultivariateNormal
from typing import Optional

from scarches.models.trvae._utils import one_hot_encoder
from scarches.models.trvae.losses import mse, zinb, nb
from lataq.trainers._utils import euclidean_dist

class EmbedCVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_sizes,
        cell_types,
        unknown_ct_names,
        conditions,
        inject_condition,
        latent_dim,
        embedding_dim,
        recon_loss,
        dr_rate, 
        beta,
        use_bn,
        use_ln,
        landmarks_labeled,
        landmarks_unlabeled,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.cell_types = cell_types
        self.n_cell_types = len(cell_types)
        self.cell_type_encoder = {
            k: v for k, v in zip(cell_types, range(len(cell_types)))
        }
        self.n_conditions = len(conditions)
        self.conditions = conditions
        self.condition_encoder = {
            k: v for k, v in zip(conditions, range(len(conditions)))
        }
        self.inject_condition = inject_condition
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_mmd = False
        self.recon_loss = recon_loss
        self.hidden_layer_sizes = hidden_layer_sizes
        self.freeze = False
        self.unknown_ct_names = unknown_ct_names
        if self.unknown_ct_names is not None:
            for unknown_ct in self.unknown_ct_names:
                self.cell_type_encoder[unknown_ct] = -1
        self.landmarks_labeled = (
            {"mean": None, "cov": None} 
            if landmarks_labeled is None 
            else landmarks_labeled
        )
        self.landmarks_unlabeled = (
            {"mean": None}
            if landmarks_unlabeled is None 
            else landmarks_unlabeled
        )
        self.new_landmarks = None

        if self.landmarks_labeled["mean"] is not None:
            # Save indices of possible new landmarks to train
            self.new_landmarks = []
            for idx in range(self.n_cell_types - len(self.landmarks_labeled["mean"])):
                self.new_landmarks.append(len(self.landmarks_labeled["mean"]) + idx)

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if recon_loss in ["nb", "zinb"]:
            self.theta = torch.nn.Parameter(torch.randn(self.input_dim, self.n_conditions))
        else:
            self.theta = None

        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.input_dim)

        self.embedding = nn.Embedding(
            self.n_conditions,
            self.embedding_dim,
        )

        print(
            "Embedding dictionary:\n", 
            f'\tNum conditions: {self.n_conditions}\n', 
            f'\tEmbedding dim: {self.embedding_dim}',
        )
        self.encoder = Encoder(
            encoder_layer_sizes,
            self.latent_dim,
            self.use_bn,
            self.use_ln,
            self.use_dr,
            self.dr_rate,
            self.embedding_dim if 'encoder' in self.inject_condition else None,
        )
        self.decoder = Decoder(
            decoder_layer_sizes,
            self.latent_dim,
            self.recon_loss,
            self.use_bn,
            self.use_ln,
            self.use_dr,
            self.dr_rate,
            self.embedding_dim if 'decoder' in self.inject_condition else None,
        )

    def forward(
        self,
        x=None,
        batch=None,
        sizefactor=None,
        celltypes=None,
        labeled=None,
    ):
        batch_embedding = self.embedding(batch)
        x_log = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_log = x
        if 'encoder' in self.inject_condition:
            z1_mean, z1_log_var = self.encoder(x_log, batch_embedding)
        else:
            z1_mean, z1_log_var = self.encoder(x_log, batch=None)
        z1 = self.sampling(z1_mean, z1_log_var)

        if 'decoder' in self.inject_condition:
            outputs = self.decoder(z1, batch_embedding)
        else:
            outputs = self.decoder(z1, batch=None)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1).mean()
        elif self.recon_loss == "zinb":
            dec_mean_gamma, dec_dropout, y1 = outputs
            size_factor_view = (
                sizefactor
                .unsqueeze(1)
                .expand(
                    dec_mean_gamma.size(0), 
                    dec_mean_gamma.size(1)
                )
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(
                one_hot_encoder(batch, self.n_conditions), 
                self.theta
            )
            dispersion = torch.exp(dispersion)
            recon_loss = -zinb(
                x=x, 
                mu=dec_mean, 
                theta=dispersion, 
                pi=dec_dropout
            ).sum(dim=-1).mean()

        elif self.recon_loss == "nb":
            dec_mean_gamma, y1 = outputs
            size_factor_view = (
                sizefactor
                .unsqueeze(1)
                .expand(
                    dec_mean_gamma.size(0), 
                    dec_mean_gamma.size(1)
                )
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(
                one_hot_encoder(batch, self.n_conditions), 
                self.theta
            )
            dispersion = torch.exp(dispersion)
            recon_loss = -nb(
                x=x, 
                mu=dec_mean, 
                theta=dispersion
            ).sum(dim=-1).mean()

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

    def add_new_cell_type(self, cell_type_name, landmarks):
        """
        Function used to add new annotation for a novel cell type.

        Parameters
        ----------
        cell_type_name: str
            Name of the new cell type
        landmarks: list
            List of indices of the unlabeled landmarks that correspond to the new cell type

        Returns
        -------
        """
        self.cell_types.append(cell_type_name)
        self.n_cell_types += 1
        self.cell_type_encoder = {k: v for k, v in zip(self.cell_types, range(len(self.cell_types)))}
        new_landmark = self.landmarks_unlabeled["mean"][landmarks].mean(0).unsqueeze(0)

        #TODO: CALCULATE COV WITH CLUSTER CORRESPONDING CELLS INSTEAD OF SETTING TO ZERO
        new_landmark_cov = torch.zeros(
            1, self.latent_dim, self.latent_dim,
            device=self.landmarks_labeled["cov"].device, requires_grad=False
        )

        self.landmarks_labeled["mean"] = torch.cat(
            (self.landmarks_labeled["mean"], new_landmark),
            dim=0
        )
        self.landmarks_labeled["cov"] = torch.cat(
            (self.landmarks_labeled["cov"], new_landmark_cov),
            dim=0
        )


    def classify(self, x, c=None, landmark=False, classes_list=None, metric="dist"):
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
            metric:
                Method to use for classification. Can be dist, gaussian, hyperbolic
            threshold:
                Threshold to use on the class probabilities to detect novel cell types,
                or mark unknown cells.


        """
        if landmark:
            latent = x
        else:
            latent = self.get_latent(x, c)

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
                cov_matrix = self.landmarks_labeled["cov"][ct_class, :]
                # ID addition for stability
                # This has to be fixed in a better way maybe
                cov_matrix = cov_matrix + torch.eye(self.latent_dim, device=cov_matrix.device) * 1e-3
                # if torch.linalg.det(cov_matrix) == 0:
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
            quantiles_view = self.landmarks_labeled["cov"].unsqueeze(0).expand(dists.size(0), dists.size(1))
            # overlap = torch.max(torch.zeros_like(dists), (quantiles_view - dists))
            # overlap = 1 - (quantiles_view - overlap / quantiles_view)
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

    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
        It is actually sampling from latent space distributions with N(mu, var), computed by encoder.
        Parameters
        ----------
        mu: torch.Tensor
                Torch Tensor of Means.
        log_var: torch.Tensor
                Torch Tensor of log. variances.
        Returns
        -------
        Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()

    def get_latent(self, x, c=None, mean=False):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
           data.
           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.
           mean: boolean
           Returns
           -------
           Returns Torch Tensor containing latent space encoding of 'x'.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        if 'encoder' in self.inject_condition:
            c = c.type(torch.cuda.LongTensor)
            embed_c = self.embedding(c)
            z_mean, z_log_var = self.encoder(x_, embed_c)
        else:
            z_mean, z_log_var = self.encoder(x_)
        latent = self.sampling(z_mean, z_log_var)
        if mean:
            return z_mean
        return latent

class Encoder(nn.Module):
    """ScArches Encoder class. Constructs the encoder sub-network of TRVAE and CVAE. It will transform primary space
       input to means and log. variances of latent space with n_dimensions = z_dimension.
       Parameters
       ----------
       layer_sizes: List
            List of first and hidden layer sizes
       latent_dim: Integer
            Bottleneck layer (z)  size.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       use_dr: Boolean
            If `True` dropout will applied to layers.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
       num_classes: Integer
            Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
            conditional VAE.
    """
    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        dr_rate: float,
        embedding_dim: int = None
    ):
        super().__init__()

        self.embedding_dim = 0
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        self.FC = None

        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
                ):
                if i == 0:
                    print(
                        "\tInput Layer in, out and cond:", 
                        in_size, 
                        out_size, 
                        self.embedding_dim
                    )
                    (
                        self
                        .FC
                        .add_module(
                            name="L{:d}".format(i), 
                            module=CondLayers(
                                in_size,
                                out_size,
                                self.embedding_dim,
                                bias=True
                            )
                        )
                    )

                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    (
                        self
                        .FC
                        .add_module(
                            name="L{:d}".format(i), 
                            module=nn.Linear(
                                in_size, 
                                out_size, 
                                bias=True
                            )
                        )
                    )
                if use_bn:
                    (
                        self
                        .FC
                        .add_module(
                            "N{:d}".format(i), 
                            module=nn.BatchNorm1d(out_size, affine=True)
                        )
                    )
                elif use_ln:
                    (
                        self
                        .FC
                        .add_module(
                            "N{:d}".format(i), 
                            module=nn.LayerNorm(
                                out_size, 
                                elementwise_affine=False
                            )
                        )
                    )
                self.FC.add_module(
                    name="A{:d}".format(i), 
                    module=nn.ReLU()
                )
                if use_dr:
                    self.FC.add_module(
                        name="D{:d}".format(i), 
                        module=nn.Dropout(p=dr_rate)
                    )

        print("\tMean/Var Layer in/out:", layer_sizes[-1], latent_dim)
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, x, batch=None):
        if batch is not None:
        #    batch = one_hot_encoder(batch, n_cls=self.n_classes)
            x = torch.cat((x, batch), dim=-1)
        if self.FC is not None:
            x = self.FC(x)
        means = self.mean_encoder(x)
        log_vars = self.log_var_encoder(x)
        return means, log_vars


class Decoder(nn.Module):
    """ScArches Decoder class. Constructs the decoder sub-network of TRVAE or CVAE networks. It will transform the
       constructed latent space to the previous space of data with n_dimensions = x_dimension.
       Parameters
       ----------
       layer_sizes: List
            List of hidden and last layer sizes
       latent_dim: Integer
            Bottleneck layer (z)  size.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       use_dr: Boolean
            If `True` dropout will applied to layers.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
       num_classes: Integer
            Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
            conditional VAE.
    """
    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        recon_loss: str,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        dr_rate: float,
        embedding_dim: int = None
    ):
        super().__init__()
        self.use_dr = use_dr
        self.recon_loss = recon_loss
        self.embedding_dim = 0
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        layer_sizes = [latent_dim] + layer_sizes
        print("Decoder Architecture:")
        # Create first Decoder layer
        self.FirstL = nn.Sequential()
        print(
            "\tFirst Layer in, out and cond: ", 
            layer_sizes[0], 
            layer_sizes[1], 
            self.embedding_dim
        )

        (
            self
            .FirstL
            .add_module(
                name="L0", 
                module=CondLayers(
                    layer_sizes[0],
                    layer_sizes[1], 
                    self.embedding_dim, 
                    bias=False
                )
            )
        )
        if use_bn:
            (
                self
                .FirstL
                .add_module(
                    "N0", 
                    module=nn.BatchNorm1d(
                        layer_sizes[1], 
                        affine=True
                    )
                )
            )
        elif use_ln:
            (
                self
                .FirstL
                .add_module(
                    "N0", 
                    module=nn.LayerNorm(
                        layer_sizes[1],
                        elementwise_affine=False
                    )
                )
            )
        (
            self
            .FirstL
            .add_module(
                name="A0", 
                module=nn.ReLU()
            )
        )
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(
                zip(
                    layer_sizes[1:-1], 
                    layer_sizes[2:]
                )
            ):
                if i+3 < len(layer_sizes):
                    print("\tHidden Layer", i+1, "in/out:", in_size, out_size)
                    (
                        self
                        .HiddenL
                        .add_module(
                            name="L{:d}".format(i+1), 
                            module=nn.Linear(
                                in_size, 
                                out_size, 
                                bias=False
                            )
                        )
                    )
                    if use_bn:
                        (
                            self
                            .HiddenL
                            .add_module(
                                "N{:d}".format(i+1), 
                                module=nn.BatchNorm1d(
                                    out_size, 
                                    affine=True
                                )
                            )
                        )
                    elif use_ln:
                        (
                            self
                            .HiddenL
                            .add_module(
                                "N{:d}".format(i + 1), 
                                module=nn.LayerNorm(
                                    out_size, 
                                    elementwise_affine=False
                                )
                            )
                        )
                    (
                        self
                        .HiddenL
                        .add_module(
                            name="A{:d}".format(i+1), 
                            module=nn.ReLU()
                        )
                    )
                    if self.use_dr:
                        (
                            self
                            .HiddenL
                            .add_module(
                                name="D{:d}".format(i+1), 
                                module=nn.Dropout(p=dr_rate)
                            )
                        )
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        if self.recon_loss == "mse":
            self.recon_decoder = nn.Sequential(
                nn.Linear(
                    layer_sizes[-2], 
                    layer_sizes[-1]
                ), 
                nn.ReLU()
            )
        if self.recon_loss == "zinb":
            # mean gamma
            self.mean_decoder = nn.Sequential(
                nn.Linear(
                    layer_sizes[-2], 
                    layer_sizes[-1]
                ), 
                nn.Softmax(dim=-1)
            )
            # dropout
            self.dropout_decoder = nn.Linear(
                layer_sizes[-2], 
                layer_sizes[-1]
            )
        if self.recon_loss == "nb":
            # mean gamma
            self.mean_decoder = nn.Sequential(
                nn.Linear(
                    layer_sizes[-2], 
                    layer_sizes[-1]
                ), 
                nn.Softmax(dim=-1)
            )

    def forward(self, z, batch=None):
        # Add Condition Labels to Decoder Input
        if batch is not None:
            #batch = one_hot_encoder(batch, n_cls=self.n_classes)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.FirstL(z_cat)
        else:
            dec_latent = self.FirstL(z)

        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        # Compute Decoder Output
        if self.recon_loss == "mse":
            recon_x = self.recon_decoder(x)
            return recon_x, dec_latent
        elif self.recon_loss == "zinb":
            dec_mean_gamma = self.mean_decoder(x)
            dec_dropout = self.dropout_decoder(x)
            return dec_mean_gamma, dec_dropout, dec_latent
        elif self.recon_loss == "nb":
            dec_mean_gamma = self.mean_decoder(x)
            return dec_mean_gamma, dec_latent

class CondLayers(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_cond: int,
            bias: bool,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            out = self.expr_L(x)
        else:
            expr, cond = torch.split(
                x, 
                [x.shape[1] - self.n_cond, self.n_cond],
                dim=1
            )
            out = self.expr_L(expr) + self.cond_L(cond)
        return out