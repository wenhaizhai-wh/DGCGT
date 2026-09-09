DGCGT

Dual branch geometric constraint graph transformer for rare cell identification in single cell expression data.

This repository contains the DGCGT source code, the simulated data generator, the real data input instructions, the manuscript figures, and the supplied result tables.

Abstract

Single cell RNA sequencing can reveal cellular heterogeneity at high resolution, but rare cell detection is affected by class imbalance, dropout events, and noise. DGCGT uses three components to address these problems. SISD performs graph based denoising. DCC extracts directional centrality from the local geometry of the cell embedding. FF combines the original features with the enhanced features in a heterogeneous graph transformer. The method is evaluated on simulated and real single cell datasets.

Code

All core programs are in codes/DGCGT. The main Python programs are run_experiments.py, run_simulated.py, generate_simulated.py, visualize_results.py, and smoke_test.py. The other Python files implement data loading, preprocessing, graph construction, model definition, training, metrics, and configuration.

Installation

Use Python 3.10 to 3.12. From the repository folder run the following commands.

python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r codes/DGCGT/requirements.txt

The complete installation and execution instructions are in codes/DGCGT/README.md.

Basic check

Run codes/DGCGT/smoke_test.py to check the installation with a small simulated dataset.

Simulated data

Run codes/DGCGT/run_simulated.py to generate and evaluate the twelve simulated datasets. Use the smoke option for a short test before running the complete experiment.

Real data

The real data are not included because of their size and data use restrictions. The required file format is described in data/README.md. The real data benchmark and ablation are run with codes/DGCGT/run_experiments.py.

Results and figures

The supplied aggregate results are in Results. The manuscript figures are in Figures. New experiment outputs should be written to a separate results folder and should not overwrite the supplied tables.

The reported metrics are F1, Precision, Recall, AUPRC, and ROC AUC. F1, Precision, and Recall use a score threshold of 0.5. The parameter settings are described in codes/DGCGT/PARAMETER_GUIDE.md.

The final author list, publication DOI, and repository license should be added before the repository is made public.
