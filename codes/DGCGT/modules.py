"""SISD and DCC feature enhancement modules."""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors

def build_knn_adjacency(x, k, sigma_sq=None, include_self=True):
    """Build a symmetric normalized weighted KNN adjacency matrix."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    k = min(int(k), max(1, n - 1))
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(x)
    distances, indices = knn.kneighbors(x)
    d2 = distances ** 2
    if sigma_sq is None:
        sigma_sq = float(np.median(d2[:, 1:])) + 1e-8
    weights = np.exp(-d2 / (2.0 * sigma_sq))
    rows = np.repeat(np.arange(n), k + 1)
    cols = indices.ravel()
    from scipy.sparse import csr_matrix

    adj = csr_matrix((weights.ravel(), (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)
    if not include_self:
        adj.setdiag(0)
        adj.eliminate_zeros()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    dinv = diags(1.0 / np.sqrt(deg))
    return (dinv @ adj @ dinv).tocsr()

def SISD_smooth(x, k=15, alpha=0.40, steps=2):
    """Apply iterative graph smoothing to the cell features."""
    x = np.asarray(x, dtype=np.float32)
    adj = build_knn_adjacency(x, k, include_self=True)
    out = x.copy()
    for _ in range(int(steps)):
        out = ((1.0 - alpha) * out + alpha * (adj @ out)).astype(np.float32)
    return out

def SISD_smooth_multiscale(x, k_list=(12, 15, 20), alpha=0.40, steps=2):
    """Average SISD smoothing results across several neighborhood sizes."""
    outs = [SISD_smooth(x, k=k, alpha=alpha, steps=steps) for k in k_list]
    return np.mean(np.stack(outs, axis=0), axis=0).astype(np.float32)

def directional_centrality(x, k=10):
    """Calculate directional centrality from local covariance entropy."""
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape
    k = min(int(k), max(1, n - 1))
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(x)
    _, indices = knn.kneighbors(x)
    neighbors = indices[:, 1:]
    dcm = np.zeros(n, dtype=np.float32)
    log_d = np.log(d)
    for i in range(n):
        nb = x[neighbors[i]]
        mu = nb.mean(axis=0)
        cov = (nb - mu).T @ (nb - mu) / max(1, k - 1)
        eig = np.clip(np.linalg.eigvalsh(cov), 1e-12, None)
        p = eig / eig.sum()
        entropy = -float(np.sum(p * np.log(p)))
        dcm[i] = float(np.clip(1.0 - entropy / log_d, 0.0, 1.0))
    return dcm

def enhance_with_dcm(x, dcm, weight=0.30):
    """Amplify features with directional centrality values."""
    x = np.asarray(x, dtype=np.float32)
    dcm = np.asarray(dcm, dtype=np.float32)

    amplified = x * (1.0 + weight * dcm[:, None])
    return np.concatenate([amplified, dcm[:, None]], axis=1).astype(np.float32)

def build_enhanced_features(cell_x, cfg, rare_fraction, use_SISD=True, use_dcc=True, SISD_piecewise=None, dcc_weight=None, dcc_k_list=None):
    """Build the enhanced DGCGT branch from SISD and DCC features."""
    raw = np.asarray(cell_x, dtype=np.float32)
    if use_SISD:
        k, alpha, steps = cfg.SISD_settings(rare_fraction) if SISD_piecewise is None else _SISD_settings_from(rare_fraction, SISD_piecewise)
        denoised = SISD_smooth_multiscale(raw, k_list=(12, 15, 20), alpha=alpha, steps=steps)
    else:
        denoised = raw
    if use_dcc:
        dcm_parts = []
        for k in tuple(dcc_k_list or getattr(cfg, "dcc_k_list", (10, 15))):
            dcm_denoised = directional_centrality(denoised, k=k)
            dcm_raw = directional_centrality(raw, k=k)
            dcm_parts.append(0.5 * (dcm_denoised + dcm_raw))
        dcm = np.mean(np.stack(dcm_parts, axis=0), axis=0)
        weight = cfg.dcc_weight if dcc_weight is None else dcc_weight
        amplified = denoised * (1.0 + weight * dcm[:, None])
    else:
        dcm = np.zeros(len(raw), dtype=np.float32)
        amplified = denoised

    delta = amplified - raw
    enhanced = np.concatenate([raw, delta, dcm[:, None]], axis=1).astype(np.float32)
    return enhanced, dcm

def _SISD_settings_from(rare_fraction, piecewise):
    """Select SISD parameters from a piecewise configuration."""
    for upper, k, alpha, steps in piecewise:
        if rare_fraction < upper:
            return int(k), float(alpha), int(steps)
    _, k, alpha, steps = piecewise[-1]
    return int(k), float(alpha), int(steps)


