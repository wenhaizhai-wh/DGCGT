"""Configuration for DGCGT experiments."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    """Store preprocessing, model, training, and evaluation settings."""

    # Preprocessing settings.
    target_sum: float = 1e4
    n_hvg: int = 200
    n_hvp: int = 100
    rna_pca_dim: int = 25
    atac_pca_dim: int = 25

    # Graph construction settings.
    knn_cell_k: int = 15
    cell_gene_neighbors: int = 12
    cell_peak_neighbors: int = 10
    gene_peak_neighbors: int = 3

    # SISD settings selected by the rare cell fraction.
    SISD_piecewise: tuple = ((0.10, 15, 0.35, 2), (0.30, 12, 0.30, 2), (1.01, 8, 0.25, 1))

    # DCC settings.
    dcc_k: int = 15
    dcc_k_list: tuple = (10, 15)
    dcc_weight: float = 0.60

    # Heterogeneous graph transformer settings.
    hidden_dim: int = 128
    heads: int = 4
    layers: int = 1
    dropout: float = 0.30

    dgcgt_hidden_dim: int = 96
    dgcgt_layers: int = 1
    dgcgt_use_gate: bool = False

    # Training settings.
    epochs: int = 30
    n_ensemble: int = 5
    neg_ratio: float = 1.0
    learning_rate: float = 1.5e-3
    lr_decay: float = 0.98
    weight_decay: float = 1e-3
    grad_clip: float = 5.0

    # Evaluation settings.
    n_splits: int = 5
    test_size: float = 0.20
    seed: int = 2026

    # Memory settings. A value of zero uses all cells.
    max_cells: int = 2000
    max_genes: int = 200
    max_peaks: int = 100

    def SISD_settings(self, rare_fraction: float):
        """Return the SISD settings for the given rare cell fraction."""
        for upper, k, alpha, steps in self.SISD_piecewise:
            if rare_fraction < upper:
                return int(k), float(alpha), int(steps)
        _, k, alpha, steps = self.SISD_piecewise[-1]
        return int(k), float(alpha), int(steps)

PER_DATASET_DGCGT = {

    # Shared settings for the simulated datasets.
    "simulated": {
        "dgcgt_hidden_dim": 128,
        "dgcgt_layers": 1,
        "dropout": 0.20,
        "dgcgt_n_ensemble": 5,
        "dgcgt_epochs": 50,
        "dcc_weight": 0.80,
        "SISD_piecewise": ((0.10, 15, 0.40, 3), (0.30, 12, 0.35, 3), (1.01, 8, 0.30, 2)),
    },
    "Mouse_retina": {
        "dgcgt_hidden_dim": 128,
        "dgcgt_layers": 1,
        "dropout": 0.30,
        "dgcgt_n_ensemble": 13,
        "dgcgt_epochs": 80,
        "dgcgt_ablation_epochs": 30,
        "dcc_weight": 3.00,
        "SISD_piecewise": ((0.10, 15, 0.10, 1), (0.30, 12, 0.10, 1), (1.01, 8, 0.10, 1)),
    },
    "B_lymphoma": {
        "dgcgt_hidden_dim": 128,
        "dgcgt_layers": 1,
        "dropout": 0.10,
        "dgcgt_n_ensemble": 13,
        "dgcgt_epochs": 60,
        "dgcgt_ablation_epochs": 40,
        "dcc_weight": 2.00,
        "SISD_piecewise": ((0.10, 15, 0.15, 1), (0.30, 12, 0.15, 1), (1.01, 8, 0.15, 1)),
    },
    "PBMCs_sampled": {
        "dgcgt_hidden_dim": 128,
        "dgcgt_layers": 1,
        "dropout": 0.20,
        "dgcgt_n_ensemble": 13,
        "dgcgt_epochs": 60,
        "dgcgt_ablation_epochs": 40,
        "dcc_weight": 3.00,
        "SISD_piecewise": ((0.10, 15, 0.10, 1), (0.30, 12, 0.10, 1), (1.01, 8, 0.10, 1)),
    },
}
