"""Small end-to-end check for a fresh DGCGT installation."""

from __future__ import annotations

from sklearn.model_selection import StratifiedShuffleSplit

from config import ExperimentConfig
from data import simulate_multiomics
from preprocess import preprocess_split
from train import evaluate_one_split

def main() -> None:
    rna, atac, links, labels, metadata = simulate_multiomics(
        seed=2026,
        geometry="boundary",
        n_cells=120,
        n_genes=40,
        n_peaks=20,
        rare_fraction=0.10,
    )
    cfg = ExperimentConfig(
        n_hvg=30,
        n_hvp=15,
        rna_pca_dim=10,
        atac_pca_dim=5,
        n_splits=1,
        seed=2026,
        max_cells=0,
        max_genes=40,
        max_peaks=20,
        dcc_k_list=(5, 8),
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=cfg.seed)
    train_idx, test_idx = next(splitter.split(rna, labels))
    pre = preprocess_split(
        rna[train_idx],
        rna[test_idx],
        atac[train_idx],
        atac[test_idx],
        cfg,
    )
    metrics = evaluate_one_split(
        pre,
        links,
        labels,
        train_idx,
        test_idx,
        cfg,
        seed=cfg.seed,
        device="cpu",
        variant="DGCGT",
        overrides={
            "dgcgt_hidden_dim": 16,
            "dgcgt_layers": 1,
            "dropout": 0.10,
            "dgcgt_n_ensemble": 1,
            "dgcgt_epochs": 2,
            "dcc_weight": 0.8,
            "dcc_k_list": (5, 8),
            "SISD_piecewise": ((1.01, 5, 0.10, 1),),
        },
    )
    print(f"Smoke test passed: {metadata}")
    print("Metrics:", ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))

if __name__ == "__main__":
    main()


