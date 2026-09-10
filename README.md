# DGCGT

Dual-branch Geometric Constraint Graph Transformer for rare cell identification in single-cell expression data.

## Overview

Rare cell identification in single-cell data is difficult because rare populations are strongly under-represented and the measurements contain dropout events and noise. DGCGT combines graph denoising, local geometric information, and heterogeneous graph attention to improve rare cell detection.

The model has three main components. Self-adaptive Iterative Smoothing Denoising, called SISD, reduces noise by smoothing cells on a weighted KNN graph. Directional Centrality Constraint, called DCC, measures local directional structure and highlights cells near cluster boundaries. Feature Fusion, called FF, combines the original PCA features with the SISD and DCC enhanced features. The fused features are processed by a Heterogeneous Graph Transformer.

This repository contains the core implementation, simulated-data generation, real-data input instructions, result tables, manuscript figures, and a research-only license. The original real datasets are not included.

## Repository contents

All core programs are kept directly in codes/DGCGT. The main files are described below.

data.py loads RNA and ATAC matrices, cell labels, and gene-peak links. It also generates simulated multi-omics data.

preprocess.py performs library-size normalization, log transformation for count-like data, variance-based feature selection, scaling, and PCA. Every transformation is fitted on the training part of each split.

graph.py builds the heterogeneous graph with cell, gene, and peak nodes and their relationships.

modules.py implements SISD denoising, directional centrality, and enhanced feature construction.

models.py implements the heterogeneous graph transformer, the DGCGT model, and the MarsGT-style comparison model.

train.py performs ensemble training, split evaluation, DGCGT ablations, MarsGT-style comparison, LOF evaluation, and Isolation Forest evaluation.

metrics.py calculates F1, Precision, Recall, AUPRC, and ROC-AUC and writes summary and paired-comparison tables.

config.py stores common settings and dataset-specific DGCGT settings.

run_experiments.py is the main program for real-data benchmarks and ablation experiments.

run_simulated.py runs the twelve simulated benchmark datasets.

generate_simulated.py writes simulated datasets to files without training a model.

analyze_results.py summarizes existing split-level result files without retraining.

visualize_results.py creates compact benchmark and ablation figures from result tables.

smoke_test.py checks the complete data, preprocessing, graph, model, and evaluation pipeline on a small generated dataset.

run_r_baselines.R is optional and evaluates FiRE, GapClust, and RaceID when the required R packages are available. Its instructions are in R_baselines_README.md.

The data format is described in data/README.md. The parameter details are described in codes/DGCGT/PARAMETER_GUIDE.md.

## Workflow

The program follows this workflow.

First, load the RNA matrix, optional ATAC matrix, gene-peak links, cell-type labels, and the rare cell definition.

Second, create a stratified training and test split. Normalize count-like data, select features, scale the data, and fit PCA using the training cells only.

Third, create a heterogeneous graph containing cell, gene, and peak nodes. The graph includes cell KNN relations, cell-gene relations, cell-peak relations, and gene-peak relations.

Fourth, construct the enhanced DGCGT branch with SISD and DCC. The original PCA cell features and enhanced features are then encoded by two graph transformer branches and fused before classification.

Finally, train an ensemble of models, average the cell scores, and evaluate the test cells. F1, Precision, and Recall use a score threshold of 0.5. AUPRC and ROC-AUC use the continuous scores.

## Requirements and installation

Python 3.10 or newer is recommended. The tested package versions are listed in codes/DGCGT/requirements.txt. The main dependencies are NumPy, pandas, SciPy, scikit-learn, PyTorch, h5py, and matplotlib.

The experiments can run on a CPU. A compatible CUDA version of PyTorch can be installed for larger datasets.

From the repository root, create an environment and install the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r codes/DGCGT/requirements.txt
```

On Linux or macOS, use the following activation command instead.

```bash
source .venv/bin/activate
```

Check the installation with the small end-to-end test.

```bash
python codes/DGCGT/smoke_test.py
```

The test should finish with the message Smoke test passed.

## Simulated benchmark

The simulated protocol contains twelve datasets. It uses three scenarios named noisy_dropout, boundary, and complementary, with seeds 2026, 2027, 2028, and 2029. Each dataset contains 1,200 cells, 200 genes, 100 peaks, and a rare-cell fraction of 0.03.

The simulated model uses 25 RNA PCA dimensions and 25 ATAC PCA dimensions, hidden dimension 128, one graph layer, four attention heads, dropout 0.20, DCC weight 0.80, DCC neighborhood sizes 10 and 15, five ensemble members, and 50 training epochs. SISD uses the protocol values 0.40 with 3 steps, 0.35 with 3 steps, or 0.30 with 2 steps according to the rare-cell fraction.

Run the full simulated benchmark with the following command.

```bash
python codes/DGCGT/run_simulated.py --out-dir results/reproduced/simulated
```

Use the smoke option for a small test run instead of the full benchmark.

```bash
python codes/DGCGT/run_simulated.py --smoke --out-dir results/reproduced/smoke
```

## Real-data input

The real datasets must be prepared separately. The data folder must contain Gene_Cell.mtx, which is a gene-by-cell RNA matrix. It should also contain Cell_names.tsv, Gene_names.tsv, and Cell_type.tsv. Each TSV file contains one item per line.

Paired RNA and ATAC data additionally require Peak_Cell.mtx, Peak_names.tsv, and Gene_Peak.mtx. Peak_Cell.mtx is a peak-by-cell ATAC matrix. Gene_Peak.mtx stores gene-peak connections.

Rare_types.tsv is optional. If present, write one rare cell type on each line. If it is absent, the program uses the two least frequent cell types. An h5ad file can provide labels when Cell_type.tsv is not available, provided that its observation table contains a supported label column.

The cell order must be identical in all cell-level files and must match the matrix columns. Gene names, peak names, and matrix dimensions must also agree.

## Formal experiment protocol

The following settings are the authoritative settings for the supplied paper experiments. All real-data experiments use five repeated stratified splits, with 80 percent of cells for training and 20 percent for testing. The split seed is 2026. Early stopping is disabled. The prediction threshold is 0.5.

Mouse_retina contains 9,383 cells, 150 selected genes, and 100 selected peaks. It uses 40 RNA PCA dimensions and 10 ATAC PCA dimensions, hidden dimension 128, one graph layer, four attention heads, dropout 0.30, DCC weight 3.0, DCC neighborhood sizes 10 and 15, SISD alpha 0.10, one SISD step, and 13 ensemble members. The rare cell definition is BC10 union HC. The full benchmark uses 80 epochs. The variants-only ablation uses 30 epochs.

B_lymphoma contains 14,085 cells, 500 selected genes, and 300 selected peaks. It uses 25 RNA PCA dimensions and 25 ATAC PCA dimensions, hidden dimension 128, one graph layer, four attention heads, dropout 0.10, DCC weight 2.0, DCC neighborhood sizes 10 and 15, SISD alpha 0.15, one SISD step, and 13 ensemble members. The rare cell definition is B(13) union CD4 T(11). The full benchmark uses 60 epochs. The variants-only ablation uses 40 epochs.

PBMCs_sampled contains 9,000 cells, 150 selected genes, and 100 selected peaks. It uses 40 RNA PCA dimensions and 10 ATAC PCA dimensions, hidden dimension 128, one graph layer, four attention heads, dropout 0.20, DCC weight 3.0, DCC neighborhood sizes 10 and 15, SISD alpha 0.10, one SISD step, and 13 ensemble members. The rare cell definition uses source cell types 11 and 12. The full benchmark uses 60 epochs. The variants-only ablation uses 40 epochs.

## Real-data commands

Run each command from the repository root. Replace the data path with the location of the prepared dataset. The commands include all protocol parameters so that the intended settings are visible and reproducible.

Mouse retina benchmark.

```powershell
python codes/DGCGT/run_experiments.py benchmark --data-dir "D:\path\to\Mouse_retina" --dataset-name Mouse_retina --out-dir results/reproduced/Mouse_retina --n-hvg 150 --n-hvp 100 --max-genes 150 --max-peaks 100 --rna-pca-dim 40 --atac-pca-dim 10 --hidden-dim 128 --dropout 0.30 --dcc-weight 3.0 --sisd-alpha 0.10 --sisd-steps 1 --ensemble 13 --epochs 80
```

B lymphoma benchmark.

```powershell
python codes/DGCGT/run_experiments.py benchmark --data-dir "D:\path\to\B_lymphoma" --dataset-name B_lymphoma --out-dir results/reproduced/B_lymphoma --n-hvg 500 --n-hvp 300 --max-genes 500 --max-peaks 300 --rna-pca-dim 25 --atac-pca-dim 25 --hidden-dim 128 --dropout 0.10 --dcc-weight 2.0 --sisd-alpha 0.15 --sisd-steps 1 --ensemble 13 --epochs 60
```

Sampled PBMC benchmark.

```powershell
python codes/DGCGT/run_experiments.py benchmark --data-dir "D:\path\to\PBMCs_sampled" --dataset-name PBMCs_sampled --out-dir results/reproduced/PBMCs_sampled --n-hvg 150 --n-hvp 100 --max-genes 150 --max-peaks 100 --rna-pca-dim 40 --atac-pca-dim 10 --hidden-dim 128 --dropout 0.20 --dcc-weight 3.0 --sisd-alpha 0.10 --sisd-steps 1 --ensemble 13 --epochs 60
```

The benchmark writes split-level and summary CSV files to the selected output folder. It evaluates DGCGT, the MarsGT-style comparison model, LOF, and Isolation Forest.

## Ablation analysis

Run the matching full benchmark before running its ablation. The ablation reads the DGCGT rows from the full benchmark split-level CSV and retrains only three variants: w/o DCC, w/o FF, and w/o SISD. The Full DGCGT rows are not retrained by the ablation command.

Mouse retina ablation.

```powershell
python codes/DGCGT/run_experiments.py ablation --data-dir "D:\path\to\Mouse_retina" --dataset-name Mouse_retina --external-full-csv results/reproduced/Mouse_retina/Mouse_retina_split_results.csv --out-dir results/reproduced/Mouse_retina_ablation --n-hvg 150 --n-hvp 100 --max-genes 150 --max-peaks 100 --rna-pca-dim 40 --atac-pca-dim 10 --hidden-dim 128 --dropout 0.30 --dcc-weight 3.0 --sisd-alpha 0.10 --sisd-steps 1 --ensemble 13 --epochs 30
```

B lymphoma ablation.

```powershell
python codes/DGCGT/run_experiments.py ablation --data-dir "D:\path\to\B_lymphoma" --dataset-name B_lymphoma --external-full-csv results/reproduced/B_lymphoma/B_lymphoma_split_results.csv --out-dir results/reproduced/B_lymphoma_ablation --n-hvg 500 --n-hvp 300 --max-genes 500 --max-peaks 300 --rna-pca-dim 25 --atac-pca-dim 25 --hidden-dim 128 --dropout 0.10 --dcc-weight 2.0 --sisd-alpha 0.15 --sisd-steps 1 --ensemble 13 --epochs 40
```

Sampled PBMC ablation.

```powershell
python codes/DGCGT/run_experiments.py ablation --data-dir "D:\path\to\PBMCs_sampled" --dataset-name PBMCs_sampled --external-full-csv results/reproduced/PBMCs_sampled/PBMCs_sampled_split_results.csv --out-dir results/reproduced/PBMCs_sampled_ablation --n-hvg 150 --n-hvp 100 --max-genes 150 --max-peaks 100 --rna-pca-dim 40 --atac-pca-dim 10 --hidden-dim 128 --dropout 0.20 --dcc-weight 3.0 --sisd-alpha 0.10 --sisd-steps 1 --ensemble 13 --epochs 40
```

The program also selects the protocol epoch count automatically when --epochs is omitted. Explicit values are recommended when documenting or reproducing a paper run.

## Results and figures

The supplied aggregate tables are in Results. New outputs should be written to results/reproduced so that the supplied tables are not overwritten.

Summarize existing split-level results with the following command.

```bash
python codes/DGCGT/analyze_results.py --out_dir results/reproduced/Mouse_retina --markdown
```

Create benchmark and ablation figures from the supplied aggregate tables with the following command.

```bash
python codes/DGCGT/visualize_results.py --results-dir Results --output-dir results/reproduced/Figures
```

## License

The source code and documentation are released under the DGCGT Research-Only Non-Commercial License in LICENSE. The license permits free use, modification, and redistribution for non-commercial research, academic, educational, and personal purposes with attribution. Commercial use is not permitted.

This is a custom research-use license and is not an OSI-approved open-source license. The manuscript, figures, third-party datasets, and third-party software may have separate rights or restrictions.

## Citation

If you use DGCGT, please cite the associated manuscript. Add the final author list and DOI after publication.
