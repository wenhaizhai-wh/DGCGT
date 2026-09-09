from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS = {
    "DGCGT": "#2166ac",
    "MarsGT": "#e08214",
    "LOF": "#d62728",
    "IsolationForest": "#2ca02c",
    "FiRE": "#9467bd",
    "GapClust": "#8c564b",
    "RaceID": "#e377c2",
    "w/o DCC": "#3a9d5d",
    "w/o FF": "#8b5aa8",
    "w/o SISD": "#c44747",
}

def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)

def add_values(ax, bars):
    for bar in bars:
        value = float(bar.get_height())
        inside = value > 0.72
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.065 if inside else min(value + 0.018, 0.97),
            f"{value:.3f}",
            ha="center",
            va="center" if inside else "bottom",
            fontsize=7.5,
            fontweight="bold",
            color="white" if inside else "black",
        )

def clipped_yerr(means: pd.Series, stds: pd.Series) -> np.ndarray:
    mean_values = means.to_numpy(dtype=float)
    std_values = stds.fillna(0).to_numpy(dtype=float)
    lower = np.minimum(std_values, mean_values)
    upper = np.minimum(std_values, 1.0 - mean_values)
    return np.vstack([lower, upper])

def plot_benchmark(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "benchmark_summary_all_methods_metrics.csv"
    if not path.exists():
        path = results_dir / "all_real_methods_metrics.csv"
    frame = pd.read_csv(path)
    if "dataset" not in frame or "f1_mean" not in frame:
        raise ValueError(f"Expected aggregate benchmark columns in {path}")
    order = ["Simulated", "Mouse_retina", "B_lymphoma", "PBMCs_sampled"]
    methods = ["DGCGT", "MarsGT", "LOF", "IsolationForest", "FiRE", "GapClust", "RaceID"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.92, wspace=0.08, hspace=0.45)
    for index, dataset in enumerate(order):
        ax = axes.ravel()[index]
        current = frame[frame["dataset"] == dataset].set_index("method").reindex(methods).dropna(subset=["f1_mean"])
        bars = ax.bar(
            np.arange(len(current)),
            current["f1_mean"],
            width=0.68,
            color=[METHOD_COLORS[name] for name in current.index],
            edgecolor="black",
            linewidth=0.7,
            yerr=clipped_yerr(current["f1_mean"], current["f1_std"]),
            capsize=3,
        )
        add_values(ax, bars)
        ax.set_title(dataset.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(current)), current.index, rotation=38, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis="y", labelsize=9)
        style_axes(ax)
        ax.text(0.02, 1.03, "abcd"[index], transform=ax.transAxes, fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("F1", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("F1", fontsize=11, fontweight="bold")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "benchmark_f1_2x2.png", dpi=300)
    plt.close(fig)

def plot_ablation(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "ablation_latest_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    methods = ["DGCGT", "w/o DCC", "w/o FF", "w/o SISD"]
    datasets = ["Simulated", "Mouse_retina", "B_lymphoma", "PBMCs_sampled"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=300, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.91, wspace=0.10, hspace=0.42)
    for index, dataset in enumerate(datasets):
        ax = axes.ravel()[index]
        current = frame[frame["dataset"] == dataset].set_index("method").reindex(methods).dropna(subset=["f1_mean"])
        bars = ax.bar(
            np.arange(len(current)), current["f1_mean"], width=0.60,
            color=[METHOD_COLORS[name] for name in current.index], edgecolor="black", linewidth=0.7,
            yerr=clipped_yerr(current["f1_mean"], current["f1_std"]), capsize=3,
        )
        add_values(ax, bars)
        ax.set_title(dataset.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(current)), current.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="y", labelsize=9)
        style_axes(ax)
        ax.text(0.02, 1.03, "abcd"[index], transform=ax.transAxes, fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("F1", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("F1", fontsize=11, fontweight="bold")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ablation_f1_2x2.png", dpi=300)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact benchmark and ablation F1 figures.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.results_dir.parent / "Figures_generated")
    plot_benchmark(args.results_dir, output_dir)
    plot_ablation(args.results_dir, output_dir)
    print(f"Figures written to {output_dir.resolve()}")

if __name__ == "__main__":
    main()


