import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.stats import itemfreq, entropy
from scipy.sparse.csgraph import connected_components
import sklearn
import sklearn.metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import subprocess
import tempfile
from os import mkdir, path, remove, stat

# Define Errors
class RootCellError(Exception):
    def __init__(self, message):
        self.message = message


class NeighborsError(Exception):
    def __init__(self, message):
        self.message = message


def remove_sparsity(adata):
    """
    If ``adata.X`` is a sparse matrix, this will convert it in to normal matrix.
    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    Returns
    -------
    adata: :class:`~anndata.AnnData`
        Annotated dataset.
    """
    if sparse.issparse(adata.X):
        new_adata = sc.AnnData(
            X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True)
        )
        return new_adata

    return adata


def opt_louvain(
    adata,
    label_key,
    cluster_key,
    function=None,
    resolutions=None,
    use_rep=None,
    inplace=True,
    plot=False,
    force=True,
    verbose=True,
    **kwargs,
):
    """
    params:
        label_key: name of column in adata.obs containing biological labels to be
            optimised against
        cluster_key: name of column to be added to adata.obs during clustering.
            Will be overwritten if exists and `force=True`
        function: function that computes the cost to be optimised over. Must take as
            arguments (adata, group1, group2, **kwargs) and returns a number for maximising
        resolutions: list if resolutions to be optimised over. If `resolutions=None`,
            default resolutions of 20 values ranging between 0.1 and 2 will be used
        use_rep: key of embedding to use only if adata.uns['neighbors'] is not defined,
            otherwise will be ignored
    returns:
        res_max: resolution of maximum score
        score_max: maximum score
        score_all: `pd.DataFrame` containing all scores at resolutions. Can be used to plot the score profile.
        clustering: only if `inplace=False`, return cluster assignment as `pd.Series`
        plot: if `plot=True` plot the score profile over resolution
    """

    if function is None:
        function = metrics.nmi

    if cluster_key in adata.obs.columns:
        if force:
            if verbose:
                print(
                    f"Warning: cluster key {cluster_key} already exists "
                    + "in adata.obs and will be overwritten"
                )
        else:
            raise ValueError(
                f"cluster key {cluster_key} already exists in "
                + "adata, please remove the key or choose a different name."
                + "If you want to force overwriting the key, specify `force=True`"
            )

    if resolutions is None:
        n = 20
        resolutions = [2 * x / n for x in range(1, n + 1)]

    score_max = 0
    res_max = resolutions[0]
    clustering = None
    score_all = []

    try:
        adata.uns["neighbors"]
    except KeyError:
        if verbose:
            print("computing neigbours for opt_cluster")
        sc.pp.neighbors(adata, use_rep=use_rep)

    for res in resolutions:
        sc.tl.louvain(adata, resolution=res, key_added=cluster_key)
        score = function(adata, label_key, cluster_key, **kwargs)
        score_all.append(score)
        if score_max < score:
            score_max = score
            res_max = res
            clustering = adata.obs[cluster_key]
        del adata.obs[cluster_key]

    if verbose:
        print(f"optimised clustering against {label_key}")
        print(f"optimal cluster resolution: {res_max}")
        print(f"optimal score: {score_max}")

    score_all = pd.DataFrame(
        zip(resolutions, score_all), columns=("resolution", "score")
    )
    if plot:
        # score vs. resolution profile
        sns.lineplot(data=score_all, x="resolution", y="score").set_title(
            "Optimal cluster resolution profile"
        )
        plt.show()

    if inplace:
        adata.obs[cluster_key] = clustering
        return res_max, score_max, score_all
    else:
        return res_max, score_max, score_all, clustering


### Silhouette score
def silhouette(adata, group_key, embed, metric="euclidean", scale=True):
    """
    wrapper for sklearn silhouette function values range from [-1, 1] with 1 being an ideal fit, 0 indicating
    overlapping clusters and -1 indicating misclassified cells
    :param group_key: key in adata.obs of cell labels
    :param embed: embedding key in adata.obsm, default: 'X_pca'
    """
    if embed not in adata.obsm.keys():
        sc.pp.pca(adata)
        # print(adata.obsm.keys())
        # raise KeyError(f'{embed} not in obsm')
    asw = sklearn.metrics.silhouette_score(
        X=adata.obsm[embed], labels=adata.obs[group_key], metric=metric
    )
    if scale:
        asw = (asw + 1) / 2
    return asw


def silhouette_batch(
    adata, batch_key, group_key, embed, metric="euclidean", verbose=True, scale=True
):
    """
    Silhouette score of batch labels subsetted for each group.
    params:
        batch_key: batches to be compared against
        group_key: group labels to be subsetted by e.g. cell type
        embed: name of column in adata.obsm
        metric: see sklearn silhouette score
    returns:
        all scores: absolute silhouette scores per group label
        group means: if `mean=True`
    """
    if embed not in adata.obsm.keys():
        sc.pp.pca(adata)
        # print(adata.obsm.keys())
        # raise KeyError(f'{embed} not in obsm')

    sil_all = pd.DataFrame(columns=["group", "silhouette_score"])

    for group in adata.obs[group_key].unique():
        adata_group = adata[adata.obs[group_key] == group]
        n_batches = adata_group.obs[batch_key].nunique()
        if (n_batches == 1) or (n_batches == adata_group.shape[0]):
            continue
        sil_per_group = sklearn.metrics.silhouette_samples(
            adata_group.obsm[embed], adata_group.obs[batch_key], metric=metric
        )
        # take only absolute value
        sil_per_group = [abs(i) for i in sil_per_group]
        if scale:
            # scale s.t. highest number is optimal
            sil_per_group = [1 - i for i in sil_per_group]
        d = pd.DataFrame(
            {"group": [group] * len(sil_per_group), "silhouette_score": sil_per_group}
        )
        sil_all = sil_all.append(d)
    sil_all = sil_all.reset_index(drop=True)
    sil_means = sil_all.groupby("group").mean()

    if verbose:
        print(f"mean silhouette per cell: {sil_means}")
    return sil_all, sil_means


### NMI normalised mutual information
def nmi(adata, group1, group2, method="arithmetic", nmi_dir=None):
    """
    Normalized mutual information NMI based on 2 different cluster assignments `group1` and `group2`
    params:
        adata: Anndata object
        group1: column name of `adata.obs` or group assignment
        group2: column name of `adata.obs` or group assignment
        method: NMI implementation
            'max': scikit method with `average_method='max'`
            'min': scikit method with `average_method='min'`
            'geometric': scikit method with `average_method='geometric'`
            'arithmetic': scikit method with `average_method='arithmetic'`
            'Lancichinetti': implementation by A. Lancichinetti 2009 et al.
            'ONMI': implementation by Aaron F. McDaid et al. (https://github.com/aaronmcdaid/Overlapping-NMI) Hurley 2011
        nmi_dir: directory of compiled C code if 'Lancichinetti' or 'ONMI' are specified as `method`. Compilation should be done as specified in the corresponding README.
    return:
        normalized mutual information (NMI)
    """

    if isinstance(group1, str):
        group1 = adata.obs[group1].tolist()
    elif isinstance(group1, pd.Series):
        group1 = group1.tolist()

    if isinstance(group2, str):
        group2 = adata.obs[group2].tolist()
    elif isinstance(group2, pd.Series):
        group2 = group2.tolist()

    if len(group1) != len(group2):
        raise ValueError(
            f"different lengths in group1 ({len(group1)}) and group2 ({len(group2)})"
        )

    # choose method
    if method in ["max", "min", "geometric", "arithmetic"]:
        nmi_value = sklearn.metrics.normalized_mutual_info_score(
            group1, group2, average_method=method
        )
    elif method == "Lancichinetti":
        nmi_value = nmi_Lanc(group1, group2, nmi_dir=nmi_dir)
    elif method == "ONMI":
        nmi_value = onmi(group1, group2, nmi_dir=nmi_dir)
    else:
        raise ValueError(f"Method {method} not valid")

    return nmi_value


def onmi(group1, group2, nmi_dir=None, verbose=True):
    """
    Based on implementation https://github.com/aaronmcdaid/Overlapping-NMI
    publication: Aaron F. McDaid, Derek Greene, Neil Hurley 2011
    params:
        nmi_dir: directory of compiled C code
    """

    if nmi_dir is None:
        raise FileNotFoundError(
            "Please provide the directory of the compiled C code from https://sites.google.com/site/andrealancichinetti/mutual3.tar.gz"
        )

    group1_file = write_tmp_labels(group1, to_int=False)
    group2_file = write_tmp_labels(group2, to_int=False)

    nmi_call = subprocess.Popen(
        [nmi_dir + "onmi", group1_file, group2_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    stdout, stderr = nmi_call.communicate()
    if stderr:
        print(stderr)

    nmi_out = stdout.decode()
    if verbose:
        print(nmi_out)

    nmi_split = [x.strip().split("\t") for x in nmi_out.split("\n")]
    nmi_max = float(nmi_split[0][1])

    # remove temporary files
    remove(group1_file)
    remove(group2_file)

    return nmi_max


def nmi_Lanc(group1, group2, nmi_dir="external/mutual3/", verbose=True):
    """
    paper by A. Lancichinetti 2009
    https://sites.google.com/site/andrealancichinetti/mutual
    recommended by Malte
    """

    if nmi_dir is None:
        raise FileNotFoundError(
            "Please provide the directory of the compiled C code from https://sites.google.com/site/andrealancichinetti/mutual3.tar.gz"
        )

    group1_file = write_tmp_labels(group1, to_int=False)
    group2_file = write_tmp_labels(group2, to_int=False)

    nmi_call = subprocess.Popen(
        [nmi_dir + "mutual", group1_file, group2_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    stdout, stderr = nmi_call.communicate()
    if stderr:
        print(stderr)
    nmi_out = stdout.decode().strip()

    return float(nmi_out.split("\t")[1])


def write_tmp_labels(group_assignments, to_int=False, delim="\n"):
    """
    write the values of a specific obs column into a temporary file in text format
    needed for external C NMI implementations (onmi and nmi_Lanc functions), because they require files as input
    params:
        to_int: rename the unique column entries by integers in range(1,len(group_assignments)+1)
    """
    if to_int:
        label_map = {}
        i = 1
        for label in set(group_assignments):
            label_map[label] = i
            i += 1
        labels = delim.join([str(label_map[name]) for name in group_assignments])
    else:
        labels = delim.join([str(name) for name in group_assignments])

    clusters = {label: [] for label in set(group_assignments)}
    for i, label in enumerate(group_assignments):
        clusters[label].append(str(i))

    output = "\n".join([" ".join(c) for c in clusters.values()])
    output = str.encode(output)

    # write to file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(output)
        filename = f.name

    return filename


### ARI adjusted rand index
def ari(adata, group1, group2):
    """
    params:
        adata: anndata object
        group1: ground-truth cluster assignments (e.g. cell type labels)
        group2: "predicted" cluster assignments
    The function is symmetric, so group1 and group2 can be switched
    """

    if isinstance(group1, str):
        group1 = adata.obs[group1].tolist()
    elif isinstance(group1, pd.Series):
        group1 = group1.tolist()

    if isinstance(group2, str):
        group2 = adata.obs[group2].tolist()
    elif isinstance(group2, pd.Series):
        group2 = group2.tolist()

    if len(group1) != len(group2):
        raise ValueError(
            f"different lengths in group1 ({len(group1)}) and group2 ({len(group2)})"
        )

    return sklearn.metrics.cluster.adjusted_rand_score(group1, group2)


### Isolated label score
def isolated_labels(
    adata, label_key, batch_key, embed, cluster=True, n=None, all_=False, verbose=True
):
    """
    score how well labels of isolated labels are distiguished in the dataset by
        1. clustering-based approach F1 score
        2. average-width silhouette score on isolated-vs-rest label assignment
    params:
        cluster: if True, use clustering approach, otherwise use silhouette score approach
        embed: key in adata.obsm used for silhouette score if cluster=False, or
            as representation for clustering (if neighbors missing in adata)
        n: max number of batches per label for label to be considered as isolated.
            if n is integer, consider labels that are present for n batches as isolated
            if n=None, consider minimum number of batches that labels are present in
        all_: return scores for all isolated labels instead of aggregated mean
    return:
        by default, mean of scores for each isolated label
        retrieve dictionary of scores for each label if `all_` is specified
    """

    scores = {}
    isolated_labels = get_isolated_labels(adata, label_key, batch_key, n, verbose)
    for label in isolated_labels:
        score = score_isolated_label(
            adata, label_key, label, embed, cluster, verbose=verbose
        )
        scores[label] = score

    if all_:
        return scores
    return np.mean(list(scores.values()))


def get_isolated_labels(adata, label_key, batch_key, n, verbose):
    """
    get labels that are considered isolated by the number of batches
    """

    tmp = adata.obs[[label_key, batch_key]].drop_duplicates()
    batch_per_lab = tmp.groupby(label_key).agg({batch_key: "count"})

    # threshold for determining when label is considered isolated
    if n is None:
        n = batch_per_lab.min().tolist()[0]

    if verbose:
        print(f"isolated labels: no more than {n} batches per label")

    labels = batch_per_lab[batch_per_lab[batch_key] <= n].index.tolist()
    if len(labels) == 0 and verbose:
        print(f"no isolated labels with less than {n} batches")
    return labels


def score_isolated_label(
    adata,
    label_key,
    label,
    embed,
    cluster=True,
    iso_label_key="iso_label",
    verbose=False,
):
    """
    compute label score for a single label
    params:
        adata: anndata object
        label_key: key in adata.obs of isolated label type (usually cell label)
        label: value of specific isolated label e.g. cell type/identity annotation
        embed: embedding to be passed to opt_louvain, if adata.uns['neighbors'] is missing
        cluster: if True, compute clustering-based F1 score, otherwise compute
            silhouette score on grouping of isolated label vs all other remaining labels
        iso_label_key: name of key to use for cluster assignment for F1 score or
            isolated-vs-rest assignment for silhouette score
    """
    adata_tmp = adata.copy()

    def max_f1(adata, label_key, cluster_key, label, argmax=False):
        """cluster optimizing over largest F1 score of isolated label"""
        obs = adata.obs
        max_cluster = None
        max_f1 = 0
        for cluster in obs[cluster_key].unique():
            y_pred = obs[cluster_key] == cluster
            y_true = obs[label_key] == label
            f1 = sklearn.metrics.f1_score(y_pred, y_true)
            if f1 > max_f1:
                max_f1 = f1
                max_cluster = cluster
        if argmax:
            return max_cluster
        return max_f1

    if cluster:
        # F1-score on clustering
        opt_louvain(
            adata_tmp,
            label_key,
            cluster_key=iso_label_key,
            label=label,
            use_rep=embed,
            function=max_f1,
            verbose=False,
            inplace=True,
        )
        score = max_f1(adata_tmp, label_key, iso_label_key, label, argmax=False)
    else:
        # AWS score between label
        adata_tmp.obs[iso_label_key] = adata_tmp.obs[label_key] == label
        score = silhouette(adata_tmp, iso_label_key, embed)

    del adata_tmp

    if verbose:
        print(f"{label}: {score}")

    return score


def precompute_hvg_batch(adata, batch, features, n_hvg=500, save_hvg=False):
    adata_list = splitBatches(adata, batch, hvg=features)
    hvg_dir = {}
    for i in adata_list:
        sc.pp.filter_genes(i, min_cells=1)
        n_hvg_tmp = np.minimum(n_hvg, int(0.5 * i.n_vars))
        if n_hvg_tmp < n_hvg:
            print(i.obs[batch][0] + " has less than the specified number of genes")
            print("Number of genes: " + str(i.n_vars))
        hvg = sc.pp.highly_variable_genes(
            i, flavor="cell_ranger", n_top_genes=n_hvg_tmp, inplace=False
        )
        hvg_dir[i.obs[batch][0]] = i.var.index[hvg["highly_variable"]]

    if save_hvg:
        adata.uns["hvg_before"] = hvg_dir
    else:
        return hvg_dir


### Highly Variable Genes conservation
def hvg_overlap(adata_pre, adata_post, batch, n_hvg=500):
    hvg_post = adata_post.var_names

    adata_post_list = splitBatches(adata_post, batch)
    overlap = []

    if ("hvg_before" in adata_pre.uns_keys()) and (
        set(hvg_post) == set(adata_pre.var_names)
    ):
        print("Using precomputed hvgs per batch")
        hvg_pre_list = adata_pre.uns["hvg_before"]

    else:
        hvg_pre_list = precompute_hvg_batch(adata_pre, batch, hvg_post)

        for i in range(len(adata_post_list)):  # range(len(adata_pre_list)):
            sc.pp.filter_genes(
                adata_post_list[i], min_cells=1
            )  # remove genes unexpressed (otherwise hvg might break)

            # ov = list(set(adata_pre_list[i].var_names).intersection(set(hvg_pre_list[i])))
            # adata_pre_list[i] = adata_pre_list[i][:,ov]
            # adata_post_list[i] = adata_post_list[i][:,ov]
            batch_var = adata_post_list[i].obs[batch][0]

            n_hvg_tmp = len(
                hvg_pre_list[batch_var]
            )  # adata_pre.uns['n_hvg'][hvg_post]#np.minimum(n_hvg, int(0.5*adata_post_list[i].n_vars))
            print(n_hvg_tmp)
            # if n_hvg_tmp<n_hvg:
            #    print(adata_post_list[i].obs[batch][0]+' has less than the specified number of genes')
            #    print('Number of genes: '+str(adata_post_list[i].n_vars))
            # hvg_pre = sc.pp.highly_variable_genes(adata_pre_list[i], flavor='cell_ranger', n_top_genes=n_hvg_tmp, inplace=False)
            tmp_pre = hvg_pre_list[
                batch_var
            ]  # adata_pre_list[i].var.index[hvg_pre['highly_variable']]
            hvg_post = sc.pp.highly_variable_genes(
                adata_post_list[i],
                flavor="cell_ranger",
                n_top_genes=n_hvg_tmp,
                inplace=False,
            )
            tmp_post = adata_post_list[i].var.index[hvg_post["highly_variable"]]
            n_hvg_real = np.minimum(len(tmp_pre), len(tmp_post))
            overlap.append((len(set(tmp_pre).intersection(set(tmp_post)))) / n_hvg_real)
    return np.mean(overlap)


### PC Regression
def pcr_comparison(
    adata_pre, adata_post, covariate, embed=None, n_comps=50, scale=True, verbose=False
):
    """
    Compare the effect before and after integration
    Return either the difference of variance contribution before and after integration
    or a score between 0 and 1 (`scaled=True`) with 0 if the variance contribution hasn't
    changed. The larger the score, the more different the variance contributions are before
    and after integration.
    params:
        adata_pre: uncorrected adata
        adata_post: integrated adata
        embed   : if `embed=None`, use the full expression matrix (`adata.X`), otherwise
                  use the embedding provided in `adata_post.obsm[embed]`
        scale: if True, return scaled score
    return:
        difference of R2Var value of PCR
    """

    if embed == "X_pca":
        embed = None

    pcr_before = pcr(
        adata_pre,
        covariate=covariate,
        recompute_pca=True,
        n_comps=n_comps,
        verbose=verbose,
    )
    pcr_after = pcr(
        adata_post,
        covariate=covariate,
        embed=embed,
        recompute_pca=True,
        n_comps=n_comps,
        verbose=verbose,
    )

    if scale:
        score = (pcr_before - pcr_after) / pcr_before
        if score < 0:
            print("Variance contribution increased after integration!")
            print("Setting PCR comparison score to 0.")
            score = 0
        return score
    else:
        return pcr_after - pcr_before


def pcr(adata, covariate, embed=None, n_comps=50, recompute_pca=True, verbose=False):
    """
    PCR for Adata object
    Checks whether to
        + compute PCA on embedding or expression data (set `embed` to name of embedding matrix e.g. `embed='X_emb'`)
        + use existing PCA (only if PCA entry exists)
        + recompute PCA on expression matrix (default)
    params:
        adata: Anndata object
        embed   : if `embed=None`, use the full expression matrix (`adata.X`), otherwise
                  use the embedding provided in `adata_post.obsm[embed]`
        n_comps: number of PCs if PCA should be computed
        covariate: key for adata.obs column to regress against
    return:
        R2Var of PCR
    """

    if verbose:
        print(f"covariate: {covariate}")
    covariate_values = adata.obs[covariate]

    # use embedding for PCA
    if (embed is not None) and (embed in adata.obsm):
        if verbose:
            print(f"compute PCR on embedding n_comps: {n_comps}")
        return pc_regression(adata.obsm[embed], covariate_values, n_comps=n_comps)

    # use existing PCA computation
    elif (recompute_pca == False) and ("X_pca" in adata.obsm) and ("pca" in adata.uns):
        if verbose:
            print("using existing PCA")
        return pc_regression(
            adata.obsm["X_pca"], covariate_values, pca_var=adata.uns["pca"]["variance"]
        )

    # recompute PCA
    else:
        if verbose:
            print(f"compute PCA n_comps: {n_comps}")
        return pc_regression(adata.X, covariate_values, n_comps=n_comps)


def pc_regression(
    data, covariate, pca_var=None, n_comps=50, svd_solver="arpack", verbose=False
):
    """
    params:
        data: expression or PCA matrix. Will be assumed to be PCA values, if pca_sd is given
        covariate: series or list of batch assignments
        n_comps: number of PCA components for computing PCA, only when pca_sd is not given. If no pca_sd is given and n_comps=None, comute PCA and don't reduce data
        pca_var: iterable of variances for `n_comps` components. If `pca_sd` is not `None`, it is assumed that the matrix contains PCA values, else PCA is computed
    PCA is only computed, if variance contribution is not given (pca_sd).
    """

    if isinstance(data, (np.ndarray, sparse.csr_matrix)):
        matrix = data
    else:
        raise TypeError(
            f"invalid type: {data.__class__} is not a numpy array or sparse matrix"
        )

    # perform PCA if no variance contributions are given
    if pca_var is None:

        if n_comps is None or n_comps > min(matrix.shape):
            n_comps = min(matrix.shape)

        if n_comps == min(matrix.shape):
            svd_solver = "full"

        if verbose:
            print("compute PCA")
        pca = sc.tl.pca(
            matrix,
            n_comps=n_comps,
            use_highly_variable=False,
            return_info=True,
            svd_solver=svd_solver,
            copy=True,
        )
        X_pca = pca[0].copy()
        pca_var = pca[3].copy()
        del pca
    else:
        X_pca = matrix
        n_comps = matrix.shape[1]

    ## PC Regression
    if verbose:
        print("fit regression on PCs")

    # handle categorical values
    if pd.api.types.is_numeric_dtype(covariate):
        covariate = np.array(covariate).reshape(-1, 1)
    else:
        if verbose:
            print("one-hot encode categorical values")
        covariate = pd.get_dummies(covariate)

    # fit linear model for n_comps PCs
    r2 = []
    for i in range(n_comps):
        pc = X_pca[:, [i]]
        lm = sklearn.linear_model.LinearRegression()
        lm.fit(covariate, pc)
        r2_score = np.maximum(0, lm.score(covariate, pc))
        r2.append(r2_score)

    Var = pca_var / sum(pca_var) * 100
    R2Var = sum(r2 * Var) / 100

    return R2Var


### lisi score
def get_hvg_indices(adata, verbose=True):
    if "highly_variable" not in adata.var.columns:
        if verbose:
            print(
                f"No highly variable genes computed, continuing with full matrix {adata.shape}"
            )
        return np.array(range(adata.n_vars))
    return np.where((adata.var["highly_variable"] == True))[0]


def select_hvg(adata, select=True):
    if select and "highly_variable" in adata.var:
        return adata[:, adata.var["highly_variable"]].copy()
    else:
        return adata


### diffusion for connectivites matrix extension
def diffusion_conn(adata, min_k=50, copy=True, max_iterations=26):
    """
    This function performs graph diffusion on the connectivities matrix until a
    minimum number `min_k` of entries per row are non-zero.

    Note:
    Due to self-loops min_k-1 non-zero connectivies entries is actually the stopping
    criterion. This is equivalent to `sc.pp.neighbors`.

    Returns:
       The diffusion-enhanced connectivities matrix of a copy of the AnnData object
       with the diffusion-enhanced connectivities matrix is in
       `adata.uns["neighbors"]["conectivities"]`
    """
    if "neighbors" not in adata.uns:
        raise ValueError(
            "`neighbors` not in adata object. " "Please compute a neighbourhood graph!"
        )

    if "connectivities" not in adata.obsp:
        raise ValueError(
            "`connectivities` not in `adata.obsp`. "
            "Please pass an object with connectivities computed!"
        )

    T = adata.obsp["connectivities"]

    # Normalize T with max row sum
    # Note: This keeps the matrix symmetric and ensures |M| doesn't keep growing
    T = sparse.diags(1 / np.array([T.sum(1).max()] * T.shape[0])) * T

    M = T

    # Check for disconnected component
    n_comp, labs = connected_components(
        adata.obsp["connectivities"], connection="strong"
    )

    if n_comp > 1:
        tab = pd.value_counts(labs)
        small_comps = tab.index[tab < min_k]
        large_comp_mask = np.array(~pd.Series(labs).isin(small_comps))
    else:
        large_comp_mask = np.array([True] * M.shape[0])

    T_agg = T
    i = 2
    while ((M[large_comp_mask, :][:, large_comp_mask] > 0).sum(1).min() < min_k) and (
        i < max_iterations
    ):
        print(f"Adding diffusion to step {i}")
        T_agg *= T
        M += T_agg
        i += 1

    if (M[large_comp_mask, :][:, large_comp_mask] > 0).sum(1).min() < min_k:
        raise ValueError(
            "could not create diffusion connectivities matrix"
            f"with at least {min_k} non-zero entries in"
            f"{max_iterations} iterations.\n Please increase the"
            "value of max_iterations or reduce k_min.\n"
        )

    M.setdiag(0)

    if copy:
        adata_tmp = adata.copy()
        adata_tmp.uns["neighbors"].update({"diffusion_connectivities": M})
        return adata_tmp

    else:
        return M


### diffusion neighbourhood score
def diffusion_nn(adata, k, max_iterations=26):
    """
    This function generates a nearest neighbour list from a connectivities matrix
    as supplied by BBKNN or Conos. This allows us to select a consistent number
    of nearest neighbours across all methods.

    Return:
       `k_indices` a numpy.ndarray of the indices of the k-nearest neighbors.
    """
    if "neighbors" not in adata.uns:
        raise ValueError(
            "`neighbors` not in adata object. " "Please compute a neighbourhood graph!"
        )

    if "connectivities" not in adata.obsp:
        raise ValueError(
            "`connectivities` not in `adata.obsp`. "
            "Please pass an object with connectivities computed!"
        )

    T = adata.obsp["connectivities"]

    # Row-normalize T
    T = sparse.diags(1 / T.sum(1).A.ravel()) * T

    T_agg = T ** 3
    M = T + T ** 2 + T_agg
    i = 4

    while ((M > 0).sum(1).min() < (k + 1)) and (i < max_iterations):
        # note: k+1 is used as diag is non-zero (self-loops)
        print(f"Adding diffusion to step {i}")
        T_agg *= T
        M += T_agg
        i += 1

    if (M > 0).sum(1).min() < (k + 1):
        raise NeighborsError(
            f"could not find {k} nearest neighbors in {max_iterations}"
            "diffusion steps.\n Please increase max_iterations or reduce"
            " k.\n"
        )

    M.setdiag(0)
    k_indices = np.argpartition(M.A, -k, axis=1)[:, -k:]

    return k_indices


# determine root cell for trajectory conservation metric
def get_root(adata_pre, adata_post, ct_key, dpt_dim=3):

    n_components, adata_post.obs["neighborhood"] = connected_components(
        csgraph=adata_post.obsp["connectivities"], directed=False, return_labels=True
    )

    start_clust = adata_pre.obs.groupby([ct_key]).mean()["dpt_pseudotime"].idxmin()
    min_dpt = adata_pre.obs[adata_pre.obs[ct_key] == start_clust].index
    # print(min_dpt)
    max_neigh = adata_post.obs[
        adata_post.obs["neighborhood"]
        == adata_post.obs["neighborhood"].value_counts().idxmax()
    ].index
    min_dpt = [value for value in min_dpt if value in max_neigh]

    adata_post_sub = adata_post[
        adata_post.obs["neighborhood"]
        == adata_post.obs["neighborhood"].value_counts().idxmax()
    ]

    min_dpt = [adata_post_sub.obs_names.get_loc(i) for i in min_dpt]

    # compute Diffmap for adata_post
    sc.tl.diffmap(adata_post_sub)

    # determine most extreme cell in adata_post Diffmap
    min_dpt_cell = np.zeros(len(min_dpt))
    for dim in np.arange(dpt_dim):

        diffmap_mean = adata_post_sub.obsm["X_diffmap"][:, dim].mean()
        diffmap_min_dpt = adata_post_sub.obsm["X_diffmap"][min_dpt, dim]

        # choose optimum function
        if diffmap_min_dpt.mean() < diffmap_mean:
            opt = np.argmin
        else:
            opt = np.argmax
        # count opt cell
        if len(diffmap_min_dpt) == 0:
            raise RootCellError("No root cell in largest component")
        min_dpt_cell[opt(diffmap_min_dpt)] += 1

    # root cell is cell with max vote
    return min_dpt[np.argmax(min_dpt_cell)], adata_post_sub


def trajectory_conservation(adata_pre, adata_post, label_key):
    cell_subset = adata_pre.obs.index[adata_pre.obs["dpt_pseudotime"].notnull()]
    adata_pre_sub = adata_pre[cell_subset]
    adata_post_sub = adata_post[cell_subset]

    iroot, adata_post_sub2 = get_root(adata_pre_sub, adata_post_sub, label_key)
    adata_post_sub2.uns["iroot"] = iroot

    sc.tl.dpt(adata_post_sub2)
    adata_post_sub2.obs["dpt_pseudotime"][adata_post_sub2.obs["dpt_pseudotime"] > 1] = 0
    adata_post_sub.obs["dpt_pseudotime"] = 0
    adata_post_sub.obs["dpt_pseudotime"] = adata_post_sub2.obs["dpt_pseudotime"]
    adata_post_sub.obs["dpt_pseudotime"].fillna(0)
    return (
        adata_post_sub.obs["dpt_pseudotime"].corr(
            adata_pre_sub.obs["dpt_pseudotime"], "spearman"
        )
        + 1
    ) / 2


def graph_connectivity(adata_post, label_key):
    """ "
    Metric that quantifies how connected the subgraph corresponding to each batch cluster is.
    """
    if "neighbors" not in adata_post.uns:
        sc.pp.neighbors(adata_post)
        # raise KeyError('Please compute the neighborhood graph before running this '
        #               'function!')

    clust_res = []

    for ct in adata_post.obs[label_key].cat.categories:
        adata_post_sub = adata_post[
            adata_post.obs[label_key].isin([ct]),
        ]
        _, labs = connected_components(
            adata_post_sub.obsp["connectivities"], connection="strong"
        )
        tab = pd.value_counts(labs)
        clust_res.append(tab[0] / sum(tab))

    return np.mean(clust_res)


def __entropy_from_indices(indices, n_cat):
    return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)), base=n_cat)


def entropy_batch_mixing(
    adata, label_key="batch", n_neighbors=50, n_pools=50, n_samples_per_pool=100
):
    """Computes Entory of Batch mixing metric for ``adata`` given the batch column name.
    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Annotated dataset.
    label_key: str
        Name of the column which contains information about different studies in ``adata.obs`` data frame.
    n_neighbors: int
        Number of nearest neighbors.
    n_pools: int
        Number of EBM computation which will be averaged.
    n_samples_per_pool: int
        Number of samples to be used in each pool of execution.
    Returns
    -------
    score: float
        EBM score. A float between zero and one.
    """
    adata = remove_sparsity(adata)
    n_cat = len(adata.obs[label_key].unique().tolist())
    print(f"Calculating EBM with n_cat = {n_cat}")
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = neighbors.kneighbors(adata.X, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: adata.obs[label_key].values[i])(indices)

    entropies = np.apply_along_axis(
        __entropy_from_indices, axis=1, arr=batch_indices, n_cat=n_cat
    )

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean(
            [
                np.mean(
                    entropies[np.random.choice(len(entropies), size=n_samples_per_pool)]
                )
                for _ in range(n_pools)
            ]
        )
    print("EBM:", score)
    return score


def knn_purity(adata, label_key="celltype", n_neighbors=30):
    """Computes KNN Purity metric for ``adata`` given the batch column name.
    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Annotated dataset.
    label_key: str
        Name of the column which contains information about different studies in ``adata.obs`` data frame.
    n_neighbors: int
        Number of nearest neighbors.
    Returns
    -------
    score: float
        KNN purity score. A float between 0 and 1.
    """
    adata = remove_sparsity(adata)
    labels = LabelEncoder().fit_transform(adata.obs[label_key].to_numpy())

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = nbrs.kneighbors(adata.X, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: labels[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [
        np.mean(scores[labels == i]) for i in np.unique(labels)
    ]  # per cell-type purity
    knn_p_score = np.mean(res)
    print("KNN-P:", knn_p_score)
    return knn_p_score


def metrics(
    adata,
    adata_int,
    batch_key,
    label_key,
    hvg_score_=True,
    cluster_nmi=None,
    nmi_=False,
    ari_=False,
    nmi_method="arithmetic",
    nmi_dir=None,
    silhouette_=False,
    embed="X_pca",
    si_metric="euclidean",
    pcr_=False,
    verbose=False,
    isolated_labels_=False,
    n_isolated=None,
    graph_conn_=False,
    trajectory_=False,
    ebm_=False,
    knn_=False,
):
    """
    summary of all metrics for one Anndata object
    """
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = adata.obs[label_key].astype("category")
    adata_int.obs[batch_key] = adata_int.obs[batch_key].astype("category")
    adata_int.obs[label_key] = adata_int.obs[label_key].astype("category")
    # clustering
    if nmi_ or ari_:
        print("clustering...")
        cluster_key = "cluster"
        res_max, nmi_max, nmi_all = opt_louvain(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            function=nmi,
            plot=False,
            verbose=verbose,
            inplace=True,
        )
        if cluster_nmi is not None:
            nmi_all.to_csv(cluster_nmi, header=False)
            print(f"saved clustering NMI values to {cluster_nmi}")

    results = {}

    if nmi_:
        print("NMI...")
        nmi_score = nmi(
            adata_int,
            group1=cluster_key,
            group2=label_key,
            method=nmi_method,
            nmi_dir=nmi_dir,
        )
    else:
        nmi_score = np.nan
    results["NMI_cluster/label"] = nmi_score

    if ari_:
        print("ARI...")
        ari_score = ari(adata_int, group1=cluster_key, group2=label_key)
    else:
        ari_score = np.nan
    results["ARI_cluster/label"] = ari_score

    if silhouette_:
        print("silhouette score...")
        # global silhouette coefficient
        sil_global = silhouette(
            adata_int, group_key=label_key, embed=embed, metric=si_metric
        )
        # silhouette coefficient per batch
        _, sil_clus = silhouette_batch(
            adata_int,
            batch_key=batch_key,
            group_key=label_key,
            embed=embed,
            metric=si_metric,
            verbose=False,
        )
        sil_clus = sil_clus["silhouette_score"].mean()
    else:
        sil_global = np.nan
        sil_clus = np.nan
    results["ASW_label"] = sil_global
    results["ASW_label/batch"] = sil_clus

    if pcr_:
        print("PC regression...")
        pcr_score = pcr_comparison(
            adata, adata_int, embed=embed, covariate=batch_key, verbose=verbose
        )
    else:
        pcr_score = np.nan
    results["PCR_batch"] = pcr_score

    if isolated_labels_:
        print("isolated labels...")
        il_score_clus = isolated_labels(
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=True,
            n=n_isolated,
            verbose=False,
        )
        il_score_sil = (
            isolated_labels(
                adata_int,
                label_key=label_key,
                batch_key=batch_key,
                embed=embed,
                cluster=False,
                n=n_isolated,
                verbose=False,
            )
            if silhouette_
            else np.nan
        )
    else:
        il_score_clus = np.nan
        il_score_sil = np.nan
    results["isolated_label_F1"] = il_score_clus
    results["isolated_label_silhouette"] = il_score_sil

    if graph_conn_:
        print("Graph connectivity...")
        graph_conn_score = graph_connectivity(adata_int, label_key=label_key)
    else:
        graph_conn_score = np.nan
    results["graph_conn"] = graph_conn_score

    if hvg_score_:
        hvg_score = hvg_overlap(adata, adata_int, batch_key)
    else:
        hvg_score = np.nan
    results["hvg_overlap"] = hvg_score

    if trajectory_:
        print("Trajectory conservation score...")
        try:
            trajectory_score = trajectory_conservation(
                adata, adata_int, label_key=label_key
            )
        except RootCellError:
            print("No cell of root cluster in largest connected component")
            trajectory_score = 0
    else:
        trajectory_score = np.nan
    results["trajectory"] = trajectory_score

    if ebm_:
        print("Entropy batch mixing...")
        ebm = entropy_batch_mixing(adata_int, batch_key, n_neighbors=15)
    else:
        ebm = np.nan
    results["ebm"] = ebm

    if knn_:
        print("KNN purity")
        knn = knn_purity(adata_int, label_key, n_neighbors=15)
    else:
        knn = np.nan
    results["knn"] = knn

    return pd.DataFrame.from_dict(results, orient="index")
