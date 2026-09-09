"""Evaluation metrics and result summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = ("f1", "precision", "recall", "auprc", "roc_auc")

def threshold_metrics(y_true, scores, threshold=0.5):
    """Calculate threshold metrics and continuous ranking metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if y_true.sum() == 0 or len(np.unique(y_true)) < 2:
        return {metric: 0.0 for metric in METRICS}
    pred = (scores >= threshold).astype(np.int64)
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "auprc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }

def summarize(frame, metrics=METRICS):
    """Summarize the mean and standard deviation for each method."""
    summary = frame.groupby("method")[list(metrics)].agg(["mean", "std"]).reset_index()
    summary.columns = ["method"] + [
        f"{metric}_{stat}" for metric in metrics for stat in ("mean", "std")
    ]
    return summary

def paired_wilcoxon(frame, reference="DGCGT", metric="f1"):
    """Compare each method with the reference method across splits."""
    from scipy.stats import wilcoxon

    frame = frame.copy()
    if "dataset" not in frame.columns:
        frame["dataset"] = "single"
    ref = frame.loc[frame["method"] == reference].set_index(["dataset", "split"])[metric]
    rows = []
    for method in sorted(frame["method"].unique()):
        if method == reference:
            continue
        other = frame.loc[frame["method"] == method].set_index(["dataset", "split"])[metric]
        common = ref.index.intersection(other.index)
        diff = (ref.loc[common] - other.loc[common]).to_numpy(dtype=float)
        if len(diff) == 0:
            continue
        if diff.std(ddof=1) <= 1e-12:
            p_value = 1.0
        else:
            try:
                p_value = float(wilcoxon(diff).pvalue)
            except ValueError:
                p_value = 1.0
        rows.append(
            {
                "reference": reference,
                "method": method,
                "metric": metric,
                "mean_diff": float(diff.mean()),
                "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                "n_pairs": int(len(diff)),
                "wilcoxon_p": p_value,
            }
        )
    return pd.DataFrame(rows)

def summary_to_markdown(summary, metrics=METRICS, title=None):
    """Convert an aggregate result table to Markdown text."""
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "|---|" + "---|" * len(metrics)
    lines.append(header)
    lines.append(separator)
    for _, row in summary.iterrows():
        cells = [
            f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}" for metric in metrics
        ]
        lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


