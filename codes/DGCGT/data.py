"""Data loading and simulated dataset generation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import coo_matrix, csr_matrix

SCENARIOS = {
    # Three geometric settings used by the simulated benchmark.

    "noisy_dropout": {"dropout": 0.60, "gaussian": 0.50, "rare_shift": 2.4, "rare_mode": "edge"},
    "boundary": {"dropout": 0.50, "gaussian": 0.50, "rare_shift": 0.9, "rare_mode": "boundary"},
    "complementary": {"dropout": 0.60, "gaussian": 0.50, "rare_shift": 2.2, "rare_mode": "edge_complementary"},
}
SIM_SEEDS = (2026, 2027, 2028, 2029)  # Four seeds produce twelve datasets.

def _read_mtx(path: Path) -> csr_matrix:
    """Read a Matrix Market file as a sparse CSR matrix."""
    try:
        matrix = mmread(path).tocsr()
        matrix.data = matrix.data.astype(np.float32, copy=False)
        return matrix
    except (ValueError, TypeError):
        header = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int64)
        n_rows, n_cols, _ = (int(v) for v in header)
        payload = np.loadtxt(path, skiprows=2, dtype=np.float32)
        matrix = coo_matrix(
            (payload[:, 2], (payload[:, 0].astype(np.int64) - 1, payload[:, 1].astype(np.int64) - 1)),
            shape=(n_rows, n_cols),
        )
        return matrix.tocsr().astype(np.float32)

def _read_names(path: Path):
    if not path.is_file():
        return None
    return pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False).iloc[:, 0].to_numpy()

def _read_h5ad_labels(path: Path, n_cells: int):
    """Read cell type labels from the observation table of an h5ad file."""
    import h5py

    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        key = None
        for candidate in ("clu_type", "pred_type_name", "pred_type"):
            if candidate in obs:
                key = candidate
                break
        if key is None:
            raise ValueError(f"no label column in h5ad obs: {list(obs.keys())}")
        categories = obs[key]["categories"][:]
        categories = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in categories]
        codes = obs[key]["codes"][:]
        labels = np.asarray([categories[int(c)] for c in codes])
        obs_index = obs["_index"][:]
        try:
            int_index = np.asarray([int(x) for x in obs_index])
        except (TypeError, ValueError):
            int_index = None
        if int_index is not None and len(int_index) > 0 and int_index.min() >= 0 and int_index.max() < n_cells:
            full = np.full(n_cells, "unknown", dtype=object)
            mask = np.zeros(n_cells, dtype=bool)
            full[int_index] = labels
            mask[int_index] = True
            return full, mask
        if len(labels) != n_cells:
            raise ValueError(f"h5ad obs length {len(labels)} != n_cells {n_cells}")
        return labels, np.ones(n_cells, dtype=bool)

def load_multiomics(data_dir, max_cells=2000, max_genes=200, max_peaks=100):
    """Load aligned RNA, ATAC, link, and rare cell label data."""
    root = Path(data_dir)
    required = ["Gene_Cell.mtx"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {root}: {', '.join(missing)}")

    rna = _read_mtx(root / "Gene_Cell.mtx")
    cell_names = _read_names(root / "Cell_names.tsv")
    gene_names = _read_names(root / "Gene_names.tsv")
    labels = _read_names(root / "Cell_type.tsv")
    label_mask = None
    if labels is None:
        h5ad_candidates = sorted(root.glob("*.h5ad"))
        if h5ad_candidates:
            labels, label_mask = _read_h5ad_labels(h5ad_candidates[0], rna.shape[1])
        else:
            raise FileNotFoundError("Cell_type.tsv is required unless an .h5ad with obs labels is present")
    has_atac = (root / "Peak_Cell.mtx").is_file() and (root / "Gene_Peak.mtx").is_file()
    if has_atac:
        atac = _read_mtx(root / "Peak_Cell.mtx")
        links = _read_mtx(root / "Gene_Peak.mtx")
        peak_names = _read_names(root / "Peak_names.tsv")
    else:
        atac = links = peak_names = None

    if cell_names is None:
        cell_names = np.asarray([f"cell_{i}" for i in range(rna.shape[1])])
    if gene_names is None:
        gene_names = np.asarray([f"gene_{i}" for i in range(rna.shape[0])])
    if labels is None:
        raise FileNotFoundError("Cell_type.tsv is required for the rare-cell definition")
    if rna.shape[1] != len(cell_names) or len(labels) != len(cell_names):
        raise ValueError("RNA, Cell_names.tsv, and Cell_type.tsv are not cell-aligned")
    if has_atac:
        if atac.shape[1] != len(cell_names):
            raise ValueError("Peak_Cell.mtx is not cell-aligned with Gene_Cell.mtx")
        if peak_names is not None and links.shape != (len(gene_names), len(peak_names)):
            raise ValueError("Gene_Peak.mtx shape does not match names")

    n_keep = len(cell_names) if int(max_cells) <= 0 else min(int(max_cells), len(cell_names))
    if len(cell_names) > n_keep:
        rng = np.random.default_rng(2026)
        cells = np.sort(rng.choice(len(cell_names), n_keep, replace=False))
    else:
        cells = np.arange(len(cell_names))

    gene_var = np.asarray(rna.power(2).mean(axis=1)).ravel() - np.asarray(rna.mean(axis=1)).ravel() ** 2
    genes = np.argsort(gene_var)[-min(max_genes, rna.shape[0]):]
    if has_atac:
        peak_var = np.asarray(atac.power(2).mean(axis=1)).ravel() - np.asarray(atac.mean(axis=1)).ravel() ** 2
        peaks = np.argsort(peak_var)[-min(max_peaks, atac.shape[0]):]
    else:
        peaks = np.array([], dtype=np.int64)
    genes.sort()
    peaks.sort()

    rna = rna[genes][:, cells].toarray().T
    labels = labels[cells]
    if has_atac:
        atac = atac[peaks][:, cells].toarray().T
        links = links[genes][:, peaks].toarray()
    else:
        atac = None
        links = None
    if label_mask is not None:
        keep = label_mask[cells]
        rna = rna[keep]
        if has_atac:
            atac = atac[keep]
        labels = labels[keep]

    rare_file = root / "Rare_types.tsv"
    if rare_file.is_file():
        rare_types = _read_names(rare_file).tolist()
    else:
        counts = pd.Series(labels).value_counts()

        rare_types = counts.index[-2:].tolist() if len(counts) > 1 else counts.index.tolist()
    y = np.isin(labels, rare_types).astype(np.int64)
    metadata = {
        "path": str(root.resolve()),
        "modalities": "rna+atac" if has_atac else "rna_only",
        "n_cells": int(rna.shape[0]),
        "n_genes": int(rna.shape[1]),
        "n_peaks": int(atac.shape[1]) if has_atac else 0,
        "rare_types": [str(v) for v in rare_types],
        "rare_count": int(y.sum()),
    }
    return (
        rna.astype(np.float32),
        atac.astype(np.float32) if has_atac else None,
        links.astype(np.float32) if has_atac else None,
        y,
        metadata,
    )

def simulate_multiomics(
    seed,
    geometry="noisy_dropout",
    n_cells=1200,
    n_genes=200,
    n_peaks=100,
    rare_fraction=0.03,
    rare_std=0.70,
):
    """Generate a noisy simulated multiomics dataset for one setting."""
    cfg = dict(SCENARIOS[geometry])
    rng = np.random.default_rng(seed)
    n_rare = max(12, int(round(n_cells * rare_fraction)))
    n_major = n_cells - n_rare
    sizes = [n_major // 3] * 3
    sizes[0] += n_major - sum(sizes)

    gene_centers = rng.normal(0.0, 1.0, size=(3, n_genes))
    gene_centers[:, 40:] *= 0.35
    peak_centers = rng.normal(0.0, 1.0, size=(3, n_peaks))
    peak_centers[:, 20:] *= 0.35
    marker_a = np.arange(0, 12)
    marker_b = np.arange(12, 24)
    peak_a = np.arange(0, 8)
    peak_b = np.arange(8, 16)

    rna_parts, atac_parts, labels = [], [], []
    for major, size in enumerate(sizes):
        rna_parts.append(rng.normal(gene_centers[major], 1.0, size=(size, n_genes)).astype(np.float32))
        atac_parts.append(rng.normal(peak_centers[major], 1.0, size=(size, n_peaks)).astype(np.float32))
        labels.append(np.zeros(size, dtype=np.int64))

    rare_gene = gene_centers[0].copy()
    rare_peak = peak_centers[0].copy()
    mode = cfg["rare_mode"]
    shift = cfg["rare_shift"]
    if mode == "edge":

        rare_gene = 0.75 * gene_centers[0] + 0.25 * gene_centers[1]
        rare_gene[marker_a] += shift
        rare_peak = 0.75 * peak_centers[0] + 0.25 * peak_centers[1]
        rare_peak[peak_a] += 0.8 * shift
    elif mode == "edge_complementary":

        rare_gene = 0.75 * gene_centers[0] + 0.25 * gene_centers[1]
        rare_gene[marker_a] += shift
        rare_gene[marker_b] -= 0.5 * shift
        rare_peak = 0.75 * peak_centers[0] + 0.25 * peak_centers[1]
        rare_peak[peak_a] -= 0.5 * shift
        rare_peak[peak_b] += shift
    elif mode == "embedded":

        rare_gene[marker_a] += shift
        rare_peak[peak_a] += 0.8 * shift
        rare_peak[peak_b] += 0.4 * shift
    elif mode == "boundary":
        rare_gene = 0.5 * gene_centers[0] + 0.5 * gene_centers[1]
        rare_gene[marker_a] += shift
        rare_peak = 0.5 * peak_centers[0] + 0.5 * peak_centers[1]
        rare_peak[peak_a] += 0.7 * shift
    else:
        rare_gene[marker_a] += shift
        rare_gene[marker_b] -= 0.5 * shift
        rare_peak[peak_a] -= 0.5 * shift
        rare_peak[peak_b] += shift

    rna_parts.append(rng.normal(rare_gene, rare_std, size=(n_rare, n_genes)).astype(np.float32))
    atac_parts.append(rng.normal(rare_peak, rare_std, size=(n_rare, n_peaks)).astype(np.float32))
    labels.append(np.ones(n_rare, dtype=np.int64))

    rna = np.vstack(rna_parts)
    atac = np.vstack(atac_parts)
    y = np.concatenate(labels)

    rna += rng.normal(0.0, cfg["gaussian"], rna.shape).astype(np.float32)
    atac += rng.normal(0.0, cfg["gaussian"], atac.shape).astype(np.float32)
    rna[rng.random(rna.shape) < cfg["dropout"]] = 0.0
    atac[rng.random(atac.shape) < cfg["dropout"]] = 0.0

    links = np.zeros((n_genes, n_peaks), dtype=np.float32)
    for gene in range(n_genes):
        picked = rng.choice(n_peaks, size=min(3, n_peaks), replace=False)
        links[gene, picked] = rng.uniform(0.3, 1.0, size=len(picked))

    order = rng.permutation(len(y))
    metadata = {
        "seed": int(seed),
        "geometry": geometry,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_peaks": int(n_peaks),
        "rare_fraction": float(rare_fraction),
        "rare_count": int(y.sum()),
        "rare_std": float(rare_std),
        **cfg,
    }
    return (
        rna[order].astype(np.float32),
        atac[order].astype(np.float32),
        links.astype(np.float32),
        y[order].astype(np.int64),
        metadata,
    )


