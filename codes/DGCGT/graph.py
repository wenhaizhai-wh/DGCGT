"""Construct the heterogeneous cell, gene, and peak graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

NODE_TYPES = ("cell", "gene", "peak")

@dataclass
class HeteroGraph:
    """Store node features and fixed neighbor slots for the graph model."""
    x: Dict[str, torch.Tensor]  # cell (n_c, d_c), gene (n_g, 3), peak (n_p, 3)
    slots: Dict[Tuple[str, str, str], torch.Tensor]
    n_train: int

    def to(self, device: str) -> "HeteroGraph":
        return HeteroGraph(
            {key: value.to(device) for key, value in self.x.items()},
            {key: value.to(device) for key, value in self.slots.items()},
            self.n_train,
        )

def _largest_indices(values: np.ndarray, k: int):
    """Return the indices of the k largest values in each row."""
    values = np.asarray(values, dtype=np.float32)
    n, m = values.shape
    k = max(1, min(int(k), m))
    out = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        order = np.argsort(values[i], kind="stable")[::-1]
        picked = order[:k]
        picked = picked[values[i, picked] > 0]
        out[i, : len(picked)] = picked
    return out

def _reverse_slots(forward_slots: np.ndarray, n_src: int, cap: int):
    """Build reverse neighbor slots with a fixed capacity."""
    forward_slots = np.asarray(forward_slots, dtype=np.int64)
    n_dst = forward_slots.shape[0]
    lists = [[] for _ in range(n_src)]
    for dst in range(n_dst):
        for src in forward_slots[dst]:
            if src >= 0:
                lists[int(src)].append(dst)
    out = np.full((n_src, cap), -1, dtype=np.int64)
    for src in range(n_src):
        ids = np.asarray(lists[src], dtype=np.int64)[:cap]
        out[src, : len(ids)] = ids
        if len(ids) < cap:
            out[src, len(ids):] = src
    return out

def build_hetero_graph(cell_x, rna_all, atac_all, gene_peak_links, cfg, n_train, seed=0):
    """Build the heterogeneous graph for one training and test split."""
    cell_x = np.asarray(cell_x, dtype=np.float32)
    rna_all = np.asarray(rna_all, dtype=np.float32)
    n_cells, n_genes = cell_x.shape[0], rna_all.shape[1]
    has_atac = atac_all is not None and gene_peak_links is not None
    if has_atac:
        atac_all = np.asarray(atac_all, dtype=np.float32)
        gene_peak_links = np.asarray(gene_peak_links, dtype=np.float32)
        n_peaks = atac_all.shape[1]

    gene_mean = rna_all[:n_train].mean(axis=0)
    gene_var = rna_all[:n_train].var(axis=0)
    node_x = {
        "cell": cell_x,
        "gene": np.column_stack([gene_mean, gene_var, np.zeros(n_genes)]).astype(np.float32),
    }
    if has_atac:
        peak_mean = atac_all[:n_train].mean(axis=0)
        peak_var = atac_all[:n_train].var(axis=0)
        gene_degree = (gene_peak_links > 0).sum(axis=1).astype(np.float32)
        peak_degree = (gene_peak_links > 0).sum(axis=0).astype(np.float32)
        node_x["gene"] = np.column_stack(
            [gene_mean, gene_var, gene_degree / max(1, n_peaks)]
        ).astype(np.float32)
        node_x["peak"] = np.column_stack(
            [peak_mean, peak_var, peak_degree / max(1, n_genes)]
        ).astype(np.float32)

    slots = {}

    k = min(int(cfg.knn_cell_k), max(1, n_cells - 1))
    nn_indices = (
        NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        .fit(cell_x)
        .kneighbors(cell_x, return_distance=False)
    )
    slots[("cell", "knn", "cell")] = nn_indices[:, 1:]

    cg = _largest_indices(rna_all, cfg.cell_gene_neighbors)  # (n_c, 12) gene indices
    slots[("gene", "expressed_by", "cell")] = cg

    slots[("cell", "expresses", "gene")] = _reverse_slots(cg, n_genes, 30)
    if has_atac:
        cp = _largest_indices(atac_all, cfg.cell_peak_neighbors)  # (n_c, 10) peak indices
        slots[("peak", "accessible_by", "cell")] = cp
        slots[("cell", "accessible", "peak")] = _reverse_slots(cp, n_peaks, 40)
        gp = _largest_indices(gene_peak_links, cfg.gene_peak_neighbors)  # (n_g, 3) peak indices
        slots[("peak", "regulated_by", "gene")] = gp
        slots[("gene", "regulates", "peak")] = _reverse_slots(gp, n_peaks, 3)

    return HeteroGraph(
        {key: torch.from_numpy(value) for key, value in node_x.items()},
        {key: torch.from_numpy(np.asarray(value, dtype=np.int64)) for key, value in slots.items()},
        int(n_train),
    )

