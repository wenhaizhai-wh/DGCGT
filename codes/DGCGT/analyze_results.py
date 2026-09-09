"""Aggregate experiment outputs without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from metrics import paired_wilcoxon, summarize, summary_to_markdown

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"{out_dir.resolve()} does not exist")
    split_files = sorted(out_dir.glob("*_split_results.csv"))
    if not split_files:
        print(f"No *_split_results.csv found under {out_dir.resolve()}")
        return

    md_sections = []
    all_frames = []
    for path in split_files:
        frame = pd.read_csv(path)
        all_frames.append(frame)
        print(f"\n===== {path.name} =====")
        print(summarize(frame).to_string(index=False))
        print("\nPaired comparison vs DGCGT:")
        print(paired_wilcoxon(frame).to_string(index=False))
        md_sections.append((path.stem, summarize(frame)))

    combined = pd.concat(all_frames, ignore_index=True)
    print("\n===== POOLED =====")
    print(summarize(combined).to_string(index=False))
    print("\nPaired comparison vs DGCGT (pooled):")
    print(paired_wilcoxon(combined).to_string(index=False))
    md_sections.append(("POOLED", summarize(combined)))

    if args.markdown:
        lines = ["# DGCGT experiment results (analyze_results)"]
        for title, summary in md_sections:
            lines.append("")
            lines.append(summary_to_markdown(summary, title=title))
        target = out_dir / "results_table.md"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nMarkdown table written to {target.resolve()}")

if __name__ == "__main__":
    main()


