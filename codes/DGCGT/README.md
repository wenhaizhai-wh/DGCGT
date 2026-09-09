DGCGT core code

All core programs are kept in this folder. The file names describe their main purpose. No extra code folder is required.

Core data and model files

data.py reads RNA data, ATAC data, labels, and gene peak links. It also generates simulated data.

preprocess.py performs library size normalization, log transformation, feature selection, scaling, and PCA.

graph.py builds the heterogeneous graph with cell, gene, and peak nodes.

modules.py contains the SISD denoising module and the DCC directional centrality module.

models.py defines the heterogeneous graph transformer layers, the DGCGT model, and the MarsGT style comparison model.

train.py contains model training, ensemble prediction, split evaluation, and classical baseline evaluation.

metrics.py calculates F1, Precision, Recall, AUPRC, ROC AUC, summary statistics, and paired comparisons.

config.py contains the experiment parameters and dataset specific parameter settings.

Main program files

run_experiments.py is the main program for real data benchmarks and ablation experiments.

run_simulated.py is the main program for the simulated benchmark.

generate_simulated.py creates simulated input files without training a model.

analyze_results.py summarizes result files that have already been generated.

visualize_results.py creates figures from the result CSV files.

smoke_test.py runs a small end to end test of the data, graph, model, and evaluation pipeline.

run_r_baselines.R runs the optional FiRE, GapClust, and RaceID baselines. The required R package information is in R_baselines_README.md.

Important functions and classes

load_multiomics reads a real dataset. simulate_multiomics creates one simulated dataset. preprocess_split prepares one training and test split. build_hetero_graph builds the graph used by the model. build_enhanced_features creates the SISD and DCC enhanced features. DGCGTMultiOmics is the main DGCGT model. run_dataset runs the benchmark on multiple splits.

How to run

Install the packages listed in requirements.txt. Run the commands from this folder.

python smoke_test.py

python run_simulated.py --smoke

python run_experiments.py --help

The file names are kept unchanged because the programs import one another directly. Their functions and purposes are explained above.
