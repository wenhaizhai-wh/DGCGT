"""Training and evaluation routines for DGCGT benchmarks."""

from __future__ import annotations

import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import LocalOutlierFactor

from config import ExperimentConfig, PER_DATASET_DGCGT
from graph import build_hetero_graph
from metrics import summarize, threshold_metrics
from models import DGCGTMultiOmics, MarsGTLite
from modules import build_enhanced_features
from preprocess import preprocess_split

BENCHMARK_VARIANTS = ("DGCGT", "MarsGT")
ABLATION_VARIANTS = ("DGCGT", "w/o SISD", "w/o DCC", "w/o FF")
VARIANT_FLAGS = {
    "DGCGT": dict(use_SISD=True, use_dcc=True, use_ff=True),
    "w/o SISD": dict(use_SISD=False, use_dcc=True, use_ff=True),
    "w/o DCC": dict(use_SISD=True, use_dcc=False, use_ff=True),
    "w/o FF": dict(use_SISD=True, use_dcc=True, use_ff=False),
    "MarsGT": None,
}

def set_seed(seed):
    """Set the random seeds used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _fit(model_factory, graph, enhanced_cell, labels, train_mask, cfg, device, seed, n_ensemble=5, epochs=None, neg_ratio=1.0):
    """Train an ensemble and return the averaged cell scores."""
    train_cells = np.where(train_mask)[0]
    pos_cells = train_cells[labels[train_cells] == 1]
    neg_cells = train_cells[labels[train_cells] == 0]
    n_all = len(labels)
    scores = np.zeros(n_all, dtype=np.float64)
    n_ensemble = max(1, int(n_ensemble))
    epochs = int(epochs) if epochs is not None else int(cfg.epochs)
    for i in range(n_ensemble):

        set_seed(seed + i * 17)
        model = model_factory()
        model.to(device)
        graph_d = graph.to(device)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
        train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        enhanced_t = (
            torch.from_numpy(np.asarray(enhanced_cell, dtype=np.float32)).to(device)
            if enhanced_cell is not None
            else None
        )
        for _ in range(epochs):
            model.train()
            logits = model(graph_d, enhanced_t)
            if len(pos_cells) > 0 and len(neg_cells) > 0:

                n_neg = max(1, min(len(neg_cells), int(round(len(pos_cells) * float(neg_ratio)))))
                selected = np.concatenate(
                    [pos_cells, np.random.choice(neg_cells, n_neg, replace=True)]
                )
                selected_t = torch.tensor(selected, dtype=torch.long, device=device)
                batch_logits = logits[selected_t]
                batch_labels = labels_t[selected_t]
            else:
                batch_logits = logits[train_mask_t]
                batch_labels = labels_t[train_mask_t]
            loss = loss_fn(batch_logits, batch_labels)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            for group in optimizer.param_groups:
                group["lr"] *= cfg.lr_decay
        model.eval()
        with torch.no_grad():
            logits = model(graph_d, enhanced_t)
        scores += torch.sigmoid(logits).cpu().numpy()
    return scores / n_ensemble

def evaluate_one_split(pre, links, y, train_idx, test_idx, cfg, seed, device, variant, overrides=None):
    """Train one model variant and evaluate it on one test split."""
    set_seed(seed)
    od = overrides or {}
    cell_train, cell_test, rna_all, atac_all, gene_idx, peak_idx = pre
    cell_x = np.vstack([cell_train, cell_test]).astype(np.float32)
    links_sub = links[np.ix_(gene_idx, peak_idx)].astype(np.float32) if links is not None else None

    n_all = len(cell_x)
    n_train = len(train_idx)
    graph = build_hetero_graph(cell_x, rna_all, atac_all, links_sub, cfg, n_train, seed)

    train_mask = np.r_[np.ones(len(train_idx), dtype=bool), np.zeros(len(test_idx), dtype=bool)]
    labels = np.concatenate([y[train_idx], np.zeros(len(test_idx), dtype=np.int64)])

    rare_fraction = float(y[train_idx].mean())
    input_dims = {key: int(value.shape[1]) for key, value in graph.x.items()}
    if variant == "MarsGT":
        def model_factory():
            return MarsGTLite(input_dims, cfg.hidden_dim, cfg.heads, cfg.layers, cfg.dropout)

        enhanced = None
    else:
        flags = VARIANT_FLAGS[variant]
        SISD_piecewise = od.get("SISD_piecewise", cfg.SISD_piecewise)
        dcc_weight = od.get("dcc_weight", cfg.dcc_weight)
        dcc_k_list = od.get("dcc_k_list", getattr(cfg, "dcc_k_list", (10, 15)))
        enhanced, _ = build_enhanced_features(
            cell_x, cfg, rare_fraction, flags["use_SISD"], flags["use_dcc"],
            SISD_piecewise=SISD_piecewise, dcc_weight=dcc_weight, dcc_k_list=dcc_k_list,
        )
        d_hidden = od.get("dgcgt_hidden_dim", cfg.dgcgt_hidden_dim)
        d_layers = od.get("dgcgt_layers", cfg.dgcgt_layers)
        d_dropout = od.get("dropout", cfg.dropout)
        d_ens = od.get("dgcgt_n_ensemble", 5)
        d_gate = od.get("dgcgt_use_gate", cfg.dgcgt_use_gate)

        def model_factory():
            return DGCGTMultiOmics(
                input_dims,
                int(enhanced.shape[1]),
                d_hidden,
                cfg.heads,
                d_layers,
                d_dropout,
                use_ff=flags["use_ff"],
                use_gate=d_gate,
            )

    n_ens = od.get("dgcgt_n_ensemble", 5) if variant != "MarsGT" else 5
    d_epochs = od.get("dgcgt_epochs", None) if variant != "MarsGT" else None
    neg_ratio = od.get("neg_ratio", cfg.neg_ratio) if variant != "MarsGT" else cfg.neg_ratio
    scores = _fit(
        model_factory,
        graph,
        enhanced,
        labels,
        train_mask,
        cfg,
        device,
        seed,
        n_ensemble=n_ens,
        epochs=d_epochs,
        neg_ratio=neg_ratio,
    )
    return threshold_metrics(y[test_idx], scores[n_train:])

def run_dataset(rna, atac, links, y, dataset_name, cfg, out_dir, device="cpu", ablation=False, overrides=None):
    """Run the configured split benchmark and save the result tables."""
    variants = list(ABLATION_VARIANTS if ablation else BENCHMARK_VARIANTS)
    if overrides is None:
        overrides = PER_DATASET_DGCGT.get(dataset_name, {})
    records = []
    splitter = StratifiedShuffleSplit(cfg.n_splits, test_size=cfg.test_size, random_state=cfg.seed)
    for split_id, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y), 1
    ):
        atac_train = atac[train_idx] if atac is not None else None
        atac_test = atac[test_idx] if atac is not None else None
        pre = preprocess_split(rna[train_idx], rna[test_idx], atac_train, atac_test, cfg)
        for variant in variants:
            metrics = evaluate_one_split(
                pre, links, y, train_idx, test_idx, cfg, cfg.seed + split_id, device, variant,
                overrides=overrides,
            )
            records.append({"dataset": dataset_name, "split": split_id, "method": variant, **metrics})

        cell_train, cell_test = pre[0], pre[1]
        base_scores = {
            "LOF": -LocalOutlierFactor(
                n_neighbors=min(20, len(train_idx) - 1), novelty=True
            )
            .fit(cell_train)
            .score_samples(cell_test),
            "IsolationForest": -IsolationForest(random_state=cfg.seed, n_jobs=1)
            .fit(cell_train)
            .score_samples(cell_test),
        }
        for method, scores in base_scores.items():
            records.append(
                {"dataset": dataset_name, "split": split_id, "method": method, **threshold_metrics(y[test_idx], scores)}
            )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / f"{dataset_name}_split_results.csv", index=False)
    summary = summarize(frame).assign(dataset=dataset_name)
    summary.to_csv(out / f"{dataset_name}_summary.csv", index=False)
    return frame, summary


