from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from config import ExperimentConfig, PER_DATASET_DGCGT
from data import load_multiomics
from metrics import METRICS, paired_wilcoxon, summarize, summary_to_markdown
from preprocess import preprocess_split
from train import evaluate_one_split, run_dataset

VARIANTS = ("w/o DCC", "w/o FF", "w/o SISD")
DATASETS = ("Mouse_retina", "B_lymphoma", "PBMCs_sampled")

def add_data_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset-name", required=True, choices=DATASETS)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-hvg", type=int, required=True)
    parser.add_argument("--n-hvp", type=int, required=True)
    parser.add_argument("--max-genes", type=int, required=True)
    parser.add_argument("--max-peaks", type=int, required=True)
    parser.add_argument("--rna-pca-dim", type=int, required=True)
    parser.add_argument("--atac-pca-dim", type=int, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")

def add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ensemble", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--dcc-weight", type=float, required=True)
    parser.add_argument("--dcc-k-list", type=int, nargs=2, default=(10, 15))
    parser.add_argument("--sisd-alpha", type=float, required=True)
    parser.add_argument("--sisd-steps", type=int, required=True)

def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        n_splits=args.n_splits,
        seed=args.seed,
        max_cells=0,
        n_hvg=args.n_hvg,
        n_hvp=args.n_hvp,
        max_genes=args.max_genes,
        max_peaks=args.max_peaks,
        rna_pca_dim=args.rna_pca_dim,
        atac_pca_dim=args.atac_pca_dim,
    )

def load_inputs(args: argparse.Namespace):
    return load_multiomics(args.data_dir, 0, args.max_genes, args.max_peaks)

def run_benchmark(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    rna, atac, links, labels, metadata = load_inputs(args)
    print(f"Loaded {args.dataset_name}: {metadata}", flush=True)
    run_dataset(
        rna,
        atac,
        links,
        labels,
        args.dataset_name,
        cfg,
        Path(args.out_dir),
        device=args.device,
        ablation=False,
        overrides={
            **PER_DATASET_DGCGT.get(args.dataset_name, {}),
            "dgcgt_hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "dgcgt_n_ensemble": args.ensemble,
            "dgcgt_epochs": args.epochs,
            "dcc_weight": args.dcc_weight,
            "dcc_k_list": tuple(args.dcc_k_list),
            "SISD_piecewise": (
                (0.10, 15, args.sisd_alpha, args.sisd_steps),
                (0.30, 12, args.sisd_alpha, args.sisd_steps),
                (1.01, 8, args.sisd_alpha, args.sisd_steps),
            ),
        },
    )
    print(f"Benchmark results written to {Path(args.out_dir).resolve()}", flush=True)

def read_full_csv(path: Path, dataset_name: str, n_splits: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"dataset", "split", "method", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Full CSV is missing columns: {sorted(missing)}")
    frame = frame[(frame["dataset"] == dataset_name) & (frame["method"] == "DGCGT")].copy()
    frame["split"] = frame["split"].astype(int)
    if set(frame["split"]) != set(range(1, n_splits + 1)):
        raise ValueError("The external Full CSV does not contain the requested split IDs")
    return frame[["dataset", "split", "method", *METRICS]]

def run_ablation(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    external = read_full_csv(Path(args.external_full_csv), args.dataset_name, args.n_splits)
    cfg = build_config(args)
    rna, atac, links, y, metadata = load_inputs(args)
    overrides = {
        **PER_DATASET_DGCGT.get(args.dataset_name, {}),
        "dgcgt_hidden_dim": args.hidden_dim,
        "dgcgt_layers": 1,
        "dropout": args.dropout,
        "dgcgt_n_ensemble": args.ensemble,
        "dgcgt_epochs": args.epochs,
        "dcc_weight": args.dcc_weight,
        "dcc_k_list": tuple(args.dcc_k_list),
        "SISD_piecewise": (
            (0.10, 15, args.sisd_alpha, args.sisd_steps),
            (0.30, 12, args.sisd_alpha, args.sisd_steps),
            (1.01, 8, args.sisd_alpha, args.sisd_steps),
        ),
    }
    records = []
    splitter = StratifiedShuffleSplit(args.n_splits, test_size=cfg.test_size, random_state=args.seed)
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
        atac_train = atac[train_idx] if atac is not None else None
        atac_test = atac[test_idx] if atac is not None else None
        pre = preprocess_split(rna[train_idx], rna[test_idx], atac_train, atac_test, cfg)
        for variant in VARIANTS:
            metrics = evaluate_one_split(
                pre, links, y, train_idx, test_idx, cfg, cfg.seed + split_id,
                args.device, variant, overrides=overrides,
            )
            records.append({"dataset": args.dataset_name, "split": split_id, "method": variant, **metrics})
            print(f"[{args.dataset_name}] split {split_id}/{args.n_splits} {variant} finished", flush=True)
    variants = pd.DataFrame(records)
    frame = pd.concat([external, variants], ignore_index=True)
    frame.to_csv(out_dir / f"{args.dataset_name}_split_results.csv", index=False)
    summary = summarize(frame).assign(dataset=args.dataset_name)
    summary.to_csv(out_dir / f"{args.dataset_name}_summary.csv", index=False)
    paired_wilcoxon(frame).to_csv(out_dir / f"{args.dataset_name}_paired_comparison.csv", index=False)
    (out_dir / "data_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "dataset": args.dataset_name,
        "data_dir": str(Path(args.data_dir).resolve()),
        "external_full_csv": str(Path(args.external_full_csv).resolve()),
        "variants_retrained": list(VARIANTS),
        "full_retrained": False,
        "n_splits": args.n_splits,
        "ensemble": args.ensemble,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "dcc_weight": args.dcc_weight,
        "dcc_k_list": list(args.dcc_k_list),
        "sisd_alpha": args.sisd_alpha,
        "sisd_steps": args.sisd_steps,
        "metrics": list(METRICS),
    }
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (out_dir / "results_table.md").write_text(
        "# Variants-only ablation\n\n"
        f"{args.dataset_name}: {args.n_splits} stratified splits, {args.ensemble} ensemble members, "
        f"{args.epochs} fixed epochs.\n\n"
        "Only w/o DCC, w/o FF, and w/o SISD were retrained. DGCGT Full was "
        "imported from the external benchmark CSV.\n\n"
        + summary_to_markdown(summary, title=args.dataset_name)
        + "\n",
        encoding="utf-8",
    )
    print(f"Ablation results written to {out_dir.resolve()}", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run DGCGT benchmarks or variants-only ablations.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="run DGCGT, MarsGT, LOF, and IsolationForest")
    add_data_options(benchmark)
    add_model_options(benchmark)
    benchmark.set_defaults(function=run_benchmark)
    ablation = subparsers.add_parser("ablation", help="retrain only w/o DCC, w/o FF, and w/o SISD")
    add_data_options(ablation)
    add_model_options(ablation)
    ablation.add_argument("--external-full-csv", required=True)
    ablation.set_defaults(function=run_ablation)
    args = parser.parse_args()
    args.function(args)

if __name__ == "__main__":
    main()


