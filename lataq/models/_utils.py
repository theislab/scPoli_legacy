import numpy as np

def subsample_conditions(adata, cond_key, subsample):
    mask = np.full(adata.n_obs, False)
    cats = adata.obs[cond_key].unique()
    for cat in cats:
        cat_idx = np.where(adata.obs[cond_key] == cat)[0]
        size = int(len(cat_idx) * subsample)
        mask[np.random.choice(cat_idx, size, replace=False)] = True
    return adata[mask]
