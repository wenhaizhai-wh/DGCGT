# DGCGT

Dual-branch Geometric Constraint Graph Transformer for rare cell identification in single-cell expression data.

## Overview

Single-cell RNA sequencing provides high-resolution information about cellular heterogeneity. Rare cell identification is difficult because of class imbalance, dropout events, and measurement noise.

DGCGT contains three main components. The Self-adaptive Iterative Smoothing Denoising module reduces sequencing noise with graph Laplacian smoothing. The Directional Centrality Constraint module describes local geometry and extracts boundary cells in the embedding space. The Feature Fusion module combines the original PCA features with the enhanced features and processes them with a Heterogeneous Graph Transformer.

The repository contains the core implementation, simulated-data experiments, real-data input instructions, manuscript figures, and supplied result tables.

## Repository contents

`codes/DGCGT/` contains all core Python and R programs. The main entry points are:

- `smoke_test.py` checks the complete pipeline on a small simulated dataset.
- `run_simulated.py` runs the simulated benchmark.
- `generate_simulated.py` generates simulated input files.
- `run_experiments.py` runs real-data benchmarks and ablation experiments.
- `analyze_results.py` summarizes existing split-level result files.
- `visualize_results.py` creates benchmark and ablation figures.

[`codes/DGCGT/README.md`](codes/DGCGT/README.md) explains the purpose of each core source file and the main functions.

The supporting files are:

- `data/README.md` describes the required real-data file format.
- `codes/DGCGT/PARAMETER_GUIDE.md` records the main experiment settings.
- `Results/` contains the supplied aggregate result tables.
- `Figures/` contains the supplied manuscript figures.
- `Manuscript.pdf` contains the manuscript associated with this repository.

## Environment

The code requires Python 3.10, 3.11, or 3.12. The tested package versions are listed in `codes/DGCGT/requirements.txt`.

The experiments can run on a CPU. A compatible PyTorch installation and a CUDA-enabled device can be used for larger experiments by changing the `--device` option to the appropriate PyTorch device.

Full real-data and simulated benchmarks can require substantial memory and runtime. Run the smoke test before starting a full experiment.

## Installation

Run these commands from the repository root.

### 1. Clone the repository

Replace the URL with the address of your GitHub repository.

```bash
git clone https://github.com/your-account/DGCGT.git
cd DGCGT
```

### 2. Create a Python environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r codes/DGCGT/requirements.txt
```

## Reproduction workflow

### 1. Check the installation

```bash
python codes/DGCGT/smoke_test.py
```

The command should print `Smoke test passed` and a small set of evaluation metrics.

### 2. Run a simulated smoke experiment

This command is intended to verify the complete data, graph, model, training, and evaluation pipeline.

```bash
python codes/DGCGT/run_simulated.py --smoke --out-dir results/reproduced/smoke
```

### 3. Run the full simulated benchmark

The default protocol runs the twelve simulated datasets described in the manuscript.

```bash
python codes/DGCGT/run_simulated.py --out-dir results/reproduced/simulated
```

The output is written to separate folders for each simulated geometry and random seed. The number of cells, epochs, ensemble members, and other settings can be changed with `python codes/DGCGT/run_simulated.py --help`.

### 4. Prepare real data

The original real datasets are not included because of their size and data-use restrictions. Prepare the files described in [`data/README.md`](data/README.md) and keep the cell order consistent across all files.

The required files are `Gene_Cell.mtx`, `Cell_names.tsv`, `Gene_names.tsv`, and `Cell_type.tsv`. Paired RNA and ATAC data additionally require `Peak_Cell.mtx`, `Peak_names.tsv`, and `Gene_Peak.mtx`. An optional `Rare_types.tsv` file can explicitly define the rare cell types.

### 5. Run a real-data benchmark

The following is a complete Mouse retina example. Change `--data-dir` to the directory containing the prepared input files.

```bash
python codes/DGCGT/run_experiments.py benchmark --data-dir "D:\path\to\Mouse_retina" --dataset-name Mouse_retina --out-dir results/reproduced/Mouse_retina --n-hvg 150 --n-hvp 100 --max-genes 150 --max-peaks 100 --rna-pca-dim 40 --atac-pca-dim 10 --hidden-dim 128 --dropout 0.30 --dcc-weight 3.0 --sisd-alpha 0.10 --sisd-steps 1 --ensemble 13 --epochs 80
```

The available dataset names are `Mouse_retina`, `B_lymphoma`, and `PBMCs_sampled`. The recommended settings for each dataset are listed in [`PARAMETER_GUIDE.md`](codes/DGCGT/PARAMETER_GUIDE.md).

### 6. Run the ablation experiment

First run the full benchmark and use its split-level CSV as `--external-full-csv`. The ablation program retrains the variants without DCC, without FF, and without SISD.

```bash
python codes/DGCGT/run_experiments.py ablation --data-dir "D:\path\to\Mouse_retina" --dataset-name Mouse_retina --external-full-csv results/reproduced/Mouse_retina/Mouse_retina_split_results.csv --out-dir results/reproduced/Mouse_retina_ablation --n-hvg 150 --n-hvp 100 --max-genes 150 --max-peaks 100 --rna-pca-dim 40 --atac-pca-dim 10 --hidden-dim 128 --dropout 0.30 --dcc-weight 3.0 --sisd-alpha 0.10 --sisd-steps 1 --ensemble 13 --epochs 30
```

### 7. Summarize results and create figures

To summarize split-level CSV files in one output folder:

```bash
python codes/DGCGT/analyze_results.py --out_dir results/reproduced/Mouse_retina --markdown
```

To create figures from the supplied aggregate tables:

```bash
python codes/DGCGT/visualize_results.py --results-dir Results --output-dir results/reproduced/Figures
```

## Evaluation

The reported metrics are F1, Precision, Recall, AUPRC, and ROC AUC. F1, Precision, and Recall use a fixed prediction threshold of 0.5. The formal experiments use five repeated stratified splits, with 80 percent of the cells for training and 20 percent for testing. The random seed is 2026 unless it is changed on the command line.

The optional R script `codes/DGCGT/run_r_baselines.R` evaluates the FiRE, GapClust, and RaceID baselines. The required R packages and command format are described in [`R_baselines_README.md`](codes/DGCGT/R_baselines_README.md).

## License

The source code is released under the [DGCGT Research-Only Non-Commercial License](LICENSE). It allows free use, modification, and redistribution for non-commercial research, academic, educational, and personal purposes with attribution. Commercial use, including use in a commercial product, paid service, or commercial analysis, is not permitted.

This is a custom research-use license and is not an OSI-approved open-source license. The manuscript, figures, and third-party datasets may have separate rights or restrictions. Before making the repository public, replace the copyright holder information in `LICENSE` with the final author or institution name and add the publication DOI when available.

## Citation

If you use DGCGT, please cite the associated manuscript. Add the final bibliographic information and DOI to this section after publication.
