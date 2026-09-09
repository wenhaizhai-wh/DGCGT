"""Training-split preprocessing and PCA utilities."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def normalize_library_size(x, target_sum=1e4):
    """Normalize each cell to the target library size."""
    x = np.asarray(x, dtype=np.float64)
    sums = x.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return (x / sums * target_sum).astype(np.float32)

def log1p(x):
    """Apply the natural logarithm transformation."""
    return np.log1p(np.asarray(x, dtype=np.float64)).astype(np.float32)

def select_top_features(x_train, n):
    """Select the features with the largest training-set variance."""
    var = np.asarray(x_train, dtype=np.float64).var(axis=0)
    return np.argsort(var)[-int(n):]

def _transform_expression(x, target_sum):
    """Transform count-like data while preserving continuous simulated data."""
    x = np.asarray(x, dtype=np.float64)
    if float(np.max(x)) > 50.0:
        x = normalize_library_size(x, target_sum)
        x = np.log1p(x)
    return x.astype(np.float32)

def _fit_pca(train, test, n_components, seed):
    """Fit scaling and PCA on training data and apply them to both splits."""
    scaler = StandardScaler().fit(train)
    train_scaled = np.nan_to_num(scaler.transform(train), nan=0.0, posinf=0.0, neginf=0.0)
    test_scaled = np.nan_to_num(scaler.transform(test), nan=0.0, posinf=0.0, neginf=0.0)
    n_pc = max(1, min(int(n_components), train_scaled.shape[0] - 1, train_scaled.shape[1]))
    pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=seed).fit(train_scaled)
    return pca.transform(train_scaled).astype(np.float32), pca.transform(test_scaled).astype(np.float32)

def preprocess_split(rna_train, rna_test, atac_train, atac_test, cfg):
    """Preprocess one train and test split without fitting on test data."""
    rna_tr = _transform_expression(rna_train, cfg.target_sum)
    rna_te = _transform_expression(rna_test, cfg.target_sum)
    gene_idx = select_top_features(rna_tr, cfg.n_hvg)
    gene_idx.sort()
    rna_tr = rna_tr[:, gene_idx]
    rna_te = rna_te[:, gene_idx]

    rna_only = atac_train is None or atac_test is None
    if rna_only:
        cell_train, cell_test = _fit_pca(rna_tr, rna_te, 50, cfg.seed)
        return cell_train, cell_test, np.vstack([rna_tr, rna_te]), None, gene_idx, None

    atac_tr = _transform_expression(atac_train, cfg.target_sum)
    atac_te = _transform_expression(atac_test, cfg.target_sum)
    peak_idx = select_top_features(atac_tr, cfg.n_hvp)
    peak_idx.sort()
    atac_tr = atac_tr[:, peak_idx]
    atac_te = atac_te[:, peak_idx]

    if getattr(cfg, "atac_pca_dim", 25) <= 0:

        cell_train, cell_test = _fit_pca(rna_tr, rna_te, 50, cfg.seed)
    else:
        rna_pca, rna_pca_te = _fit_pca(rna_tr, rna_te, cfg.rna_pca_dim, cfg.seed)
        atac_pca, atac_pca_te = _fit_pca(atac_tr, atac_te, cfg.atac_pca_dim, cfg.seed + 1)
        cell_train = np.concatenate([rna_pca, atac_pca], axis=1).astype(np.float32)
        cell_test = np.concatenate([rna_pca_te, atac_pca_te], axis=1).astype(np.float32)
    return (
        cell_train,
        cell_test,
        np.vstack([rna_tr, rna_te]),
        np.vstack([atac_tr, atac_te]),
        gene_idx,
        peak_idx,
    )


