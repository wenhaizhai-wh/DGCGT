"""Generate the paper-aligned simulated datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import SCENARIOS, simulate_multiomics

def main():
    parser = argparse.ArgumentParser(description="Generate 12 paper-aligned simulated multi-omics datasets.")
    parser.add_argument("--out_dir", default="data/dgcgt_simulated")
    parser.add_argument("--n_cells", type=int, default=1200)
    parser.add_argument("--n_genes", type=int, default=200)
    parser.add_argument("--rare_fraction", type=float, default=0.03)
    parser.add_argument("--seed_start", type=int, default=2026)
    parser.add_argument("--seed_count", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for geometry in SCENARIOS:
        for offset in range(args.seed_count):
            seed = args.seed_start + offset
            rna, atac, links, y, metadata = simulate_multiomics(
                seed, geometry, n_cells=args.n_cells, n_genes=args.n_genes, rare_fraction=args.rare_fraction
            )
            name = f"{geometry}_{seed}"
            np.savez_compressed(
                out_dir / f"{name}.npz",
                rna=rna.astype(np.float32),
                atac=atac.astype(np.float32),
                gene_peak_links=links.astype(np.float32),
                y=y.astype(np.int64),
            )
            manifest.append({"file": f"{name}.npz", **metadata})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} simulated datasets in {out_dir.resolve()}")

if __name__ == "__main__":
    main()


