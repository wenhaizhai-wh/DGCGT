"""Run the twelve paper-aligned simulated DGCGT benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import ExperimentConfig
from data import SCENARIOS, simulate_multiomics
from train import run_dataset

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DGCGT, MarsGT-style proxy, LOF, and Isolation Forest on simulated data."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("../../results/reproduced/simulated"))
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--n-cells", type=int, default=1200)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--n-peaks", type=int, default=100)
    parser.add_argument("--rare-fraction", type=float, default=0.03)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--dcc-weight", type=float, default=0.80)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one small dataset with two splits and two epochs",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        geometries = ("boundary",)
        seed_count = 1
        n_cells, n_genes, n_peaks = 120, 40, 20
        n_splits, ensemble, epochs = 2, 1, 2
        hidden_dim, rna_pca_dim, atac_pca_dim = 16, 10, 5
        n_hvg, n_hvp = 30, 15
    else:
        geometries = tuple(SCENARIOS)
        seed_count = args.seed_count
        n_cells, n_genes, n_peaks = args.n_cells, args.n_genes, args.n_peaks
        n_splits, ensemble, epochs = args.n_splits, args.ensemble, args.epochs
        hidden_dim, rna_pca_dim, atac_pca_dim = args.hidden_dim, 25, 25
        n_hvg, n_hvp = min(200, n_genes), min(100, n_peaks)

    cfg = ExperimentConfig(
        n_hvg=n_hvg,
        n_hvp=n_hvp,
        rna_pca_dim=rna_pca_dim,
        atac_pca_dim=atac_pca_dim,
        n_splits=n_splits,
        seed=2026,
        max_cells=0,
        max_genes=n_genes,
        max_peaks=n_peaks,
        dcc_k_list=(10, 15) if n_cells > 20 else (5, 8),
    )
    dcc_k_list = cfg.dcc_k_list
    sisd_k = min(15, max(3, n_cells - 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = len(geometries) * seed_count
    completed = 0
    for geometry in geometries:
        for offset in range(seed_count):
            seed = args.seed_start + offset
            rna, atac, links, labels, metadata = simulate_multiomics(
                seed,
                geometry=geometry,
                n_cells=n_cells,
                n_genes=n_genes,
                n_peaks=n_peaks,
                rare_fraction=args.rare_fraction,
            )
            dataset_name = f"{geometry}_{seed}"
            out_dir = args.out_dir / dataset_name
            print(f"[{completed + 1}/{total}] {dataset_name}: {metadata}", flush=True)
            run_dataset(
                rna,
                atac,
                links,
                labels,
                dataset_name,
                cfg,
                out_dir,
                device=args.device,
                overrides={
                    "dgcgt_hidden_dim": hidden_dim,
                    "dgcgt_layers": 1,
                    "dropout": args.dropout,
                    "dgcgt_n_ensemble": ensemble,
                    "dgcgt_epochs": epochs,
                    "dcc_weight": args.dcc_weight,
                    "dcc_k_list": dcc_k_list,
                    "SISD_piecewise": (
                        (1.01, sisd_k, 0.40 if not args.smoke else 0.10, 3 if not args.smoke else 1),
                    ),
                },
            )
            completed += 1
    print(f"Completed {completed} simulated datasets in {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()


