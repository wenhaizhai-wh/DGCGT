# DGCGT

Dual-branch Geometric Constraint Graph Transformer for rare cell identification in single-cell expression data.

## Overview

Single-cell RNA sequencing provides high-resolution information about cellular heterogeneity. Rare cell identification is difficult because of class imbalance, dropout events, and measurement noise.

DGCGT uses three main components. Self-adaptive Iterative Smoothing Denoising reduces sequencing noise with graph Laplacian smoothing. Directional Centrality Constraint describes local geometry and identifies boundary cells in the embedding space. Feature Fusion combines the original PCA features with the enhanced features and processes them with a Heterogeneous Graph Transformer.

This repository provides:

1. The core DGCGT implementation.
2. The simulated-data benchmark.
3. Instructions for preparing real single-cell data.
4. The manuscript figures and supplied result tables.
5. A research-only non-commercial license.

## Repository structure

The main project structure is shown below.

```text
DGCGT/
README.md
LICENSE
codes/DGCGT/
data/
Figures/
Results/
Manuscript.pdf
```

All core programs are directly in codes/DGCGT. The purpose of each source file is described in codes/DGCGT/README.md.

## Workflow description

Step 1. Data preparation

Real datasets are loaded from a separate data directory. The required file format is described in data/README.md.

The required RNA files are:

Gene_Cell.mtx: gene by cell expression matrix.

Cell_names.tsv: cell names.

Gene_names.tsv: gene names.

Cell_type.tsv: cell type labels.

Paired RNA and ATAC data also require:

Peak_Cell.mtx: peak by cell ATAC matrix.

Peak_names.tsv: peak names.

Gene_Peak.mtx: connections between genes and peaks.

Rare_types.tsv is optional. It contains the rare cell types to be identified. If it is not provided, the program uses the two least frequent cell types as the rare-cell class.

Step 2. Preprocessing and graph construction

preprocess.py performs normalization, feature selection, scaling, and PCA. graph.py builds the heterogeneous graph with cell, gene, and peak nodes.

Step 3. Feature enhancement

modules.py contains the SISD denoising module and the DCC directional centrality module. The enhanced features are combined with the original PCA features in the DGCGT model.

Step 4. Model training and evaluation

models.py defines the Heterogeneous Graph Transformer and the DGCGT model. train.py performs training, ensemble prediction, split evaluation, and baseline evaluation. metrics.py calculates F1, Precision, Recall, AUPRC, and ROC AUC.

Step 5. Experiment scripts

run_simulated.py runs the simulated benchmark.

run_experiments.py runs real-data benchmarks and ablation experiments.

generate_simulated.py creates simulated input files.

analyze_results.py summarizes generated result files.

visualize_results.py creates benchmark and ablation figures.

## Software requirements

Python 3.10, 3.11, or 3.12 is required.

The tested Python packages are listed in codes/DGCGT/requirements.txt.

The main packages are numpy, pandas, scipy, scikit-learn, PyTorch, h5py, and matplotlib.

The experiments can run on a CPU. A compatible CUDA-enabled PyTorch installation can be used for larger experiments.

R is optional. It is required only for the FiRE, GapClust, and RaceID baseline script. The R requirements are described in codes/DGCGT/R_baselines_README.md.

## Quick start

Run the following commands from the repository root.

### 1. Clone the repository

Replace the URL with the address of your GitHub repository.

```bash
git clone https://github.com/your-account/DGCGT.git
cd DGCGT
```

### 2. Create the Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the packages

```bash
python -m pip install --upgrade pip
python -m pip install -r codes/DGCGT/requirements.txt
```

### 4. Check the installation

```bash
python codes/DGCGT/smoke_test.py
```

The output should contain the message Smoke test passed.

### 5. Run a simulated smoke test

```bash
python codes/DGCGT/run_simulated.py --smoke --out-dir results/reproduced/smoke
```

### 6. Run the full simulated benchmark

```bash
python codes/DGCGT/run_simulated.py --out-dir results/reproduced/simulated
```

The default protocol runs twelve simulated datasets. Use the help command to change the number of cells, epochs, ensemble members, or random seeds.

```bash
python codes/DGCGT/run_simulated.py --help
```

## Real-data benchmark

The original real datasets are not included because of their size and data-use restrictions. Prepare the input files described in data/README.md before running this analysis.

The following command is a complete Mouse retina example. Change the data directory to the location of your prepared data.

```bash
python codes/DGCGT/run_experiments.py benchmark --data-dir "D:\path\to\Mouse_retina" --dataset-name Mouse_retina --out-dir results/reproduced/Mouse_retina --n-hvg 150 --n-hvp 100 --max-genes 150 --max-peaks 100 --rna-pca-dim 40 --atac-pca-dim 10 --hidden-dim 128 --dropout 0.30 --dcc-weight 3.0 --sisd-alpha 0.10 --sisd-steps 1 --ensemble 13 --epochs 80
```

The available dataset names are Mouse_retina, B_lymphoma, and PBMCs_sampled. The recommended settings are recorded in codes/DGCGT/PARAMETER_GUIDE.md.

## Ablation analysis

Run the full benchmark first. Then use its split-level CSV file as the input for the ablation analysis.

```bash
python codes/DGCGT/run_experiments.py ablation --data-dir "D:\path\to\Mouse_retina" --dataset-name Mouse_retina --external-full-csv results/reproduced/Mouse_retina/Mouse_retina_split_results.csv --out-dir results/reproduced/Mouse_retina_ablation --n-hvg 150 --n-hvp 100 --max-genes 150 --max-peaks 100 --rna-pca-dim 40 --atac-pca-dim 10 --hidden-dim 128 --dropout 0.30 --dcc-weight 3.0 --sisd-alpha 0.10 --sisd-steps 1 --ensemble 13 --epochs 30
```

The ablation analysis retrains three variants. The variants remove DCC, FF, or SISD from DGCGT.

## Results and figures

Summarize split-level result files in one output folder with:

```bash
python codes/DGCGT/analyze_results.py --out_dir results/reproduced/Mouse_retina --markdown
```

Create benchmark and ablation figures from the supplied result tables with:

```bash
python codes/DGCGT/visualize_results.py --results-dir Results --output-dir results/reproduced/Figures
```

The supplied aggregate tables are in Results. The supplied manuscript figures are in Figures. New experiment outputs should be written to a separate folder.

## Parameters and evaluation

The main settings are recorded in codes/DGCGT/PARAMETER_GUIDE.md. Default settings are stored in config.py.

The reported metrics are F1, Precision, Recall, AUPRC, and ROC AUC. F1, Precision, and Recall use a fixed prediction threshold of 0.5. The formal experiments use five repeated stratified splits. Eighty percent of the cells are used for training and twenty percent are used for testing. The default random seed is 2026.

## License

The source code is provided under the DGCGT Research-Only Non-Commercial License in LICENSE.

The license allows free use, modification, and redistribution for non-commercial research, academic, educational, and personal purposes with attribution. Commercial use is not permitted. This includes commercial products, paid services, paid data analysis, and commercial consulting.

This is a custom research-use license and is not an OSI-approved open-source license. The manuscript, figures, and third-party datasets may have separate rights or restrictions. Replace the copyright holder information in LICENSE with the final author or institution name before making the repository public.

## Citation

If you use DGCGT, please cite the associated manuscript. Add the final author list and publication DOI after publication.
