# DGCGT parameter guide

This guide records the settings used by the supplied paper experiments. Feature selection, scaling, and PCA are fitted on the training cells in every split. The exact epoch counts are also stored in config.py.

## Training and evaluation

All real-data experiments use five repeated StratifiedShuffleSplit repetitions. Each split uses 80 percent of the cells for training and 20 percent for testing. The split seed is 2026. Early stopping is disabled. The fixed decision threshold is 0.5.

The reported metrics are F1, Precision, Recall, AUPRC, and ROC-AUC. F1, Precision, and Recall use the thresholded scores. AUPRC and ROC-AUC use the continuous cell scores.

## Real-data settings

Mouse_retina uses 150 genes, 100 peaks, 40 RNA PCA dimensions, and 10 ATAC PCA dimensions. It uses hidden dimension 128, one graph layer, four attention heads, dropout 0.30, DCC weight 3.0, DCC neighborhood sizes 10 and 15, SISD alpha 0.10, one SISD step, and 13 ensemble members. The full benchmark uses 80 epochs. Its variants-only ablation uses 30 epochs.

B_lymphoma uses 500 genes, 300 peaks, 25 RNA PCA dimensions, and 25 ATAC PCA dimensions. It uses hidden dimension 128, one graph layer, four attention heads, dropout 0.10, DCC weight 2.0, DCC neighborhood sizes 10 and 15, SISD alpha 0.15, one SISD step, and 13 ensemble members. The full benchmark uses 60 epochs. Its variants-only ablation uses 40 epochs.

PBMCs_sampled uses 150 genes, 100 peaks, 40 RNA PCA dimensions, and 10 ATAC PCA dimensions. It uses hidden dimension 128, one graph layer, four attention heads, dropout 0.20, DCC weight 3.0, DCC neighborhood sizes 10 and 15, SISD alpha 0.10, one SISD step, and 13 ensemble members. The full benchmark uses 60 epochs. Its variants-only ablation uses 40 epochs.

Mouse_retina identifies BC10 union HC as rare. B_lymphoma identifies B(13) union CD4 T(11) as rare. PBMCs_sampled identifies source cell types 11 and 12 as rare.

## Simulated-data settings

The simulated benchmark contains twelve datasets from three scenarios: noisy_dropout, boundary, and complementary. It uses seeds 2026, 2027, 2028, and 2029. Each dataset has 1,200 cells, 200 genes, 100 peaks, and a rare-cell fraction of 0.03.

The simulated data use 200 genes, 100 peaks, 25 RNA PCA dimensions, and 25 ATAC PCA dimensions. They use hidden dimension 128, one graph layer, four attention heads, dropout 0.20, DCC weight 0.80, DCC neighborhood sizes 10 and 15, five ensemble members, and 50 epochs. SISD uses the piecewise settings alpha 0.40 with 3 steps, alpha 0.35 with 3 steps, or alpha 0.30 with 2 steps according to the rare-cell fraction.

## Preprocessing and graph

Count-like matrices are library-size normalized to a target sum of 10,000 and log transformed. Continuous simulated matrices are not log transformed. Features are selected by training-set variance.

The cell graph uses 15 nearest neighbors. The heterogeneous graph uses up to 12 cell-to-gene neighbors, 10 cell-to-peak neighbors, and 3 gene-to-peak neighbors. Reverse relations use fixed capacities of 30 cell-to-gene slots, 40 cell-to-peak slots, and 3 peak-to-gene slots.

## Model components

SISD builds weighted KNN graphs with multiscale neighborhoods 12, 15, and 20, then applies iterative graph smoothing. The alpha and step count are selected from the dataset protocol and rare-cell fraction.

DCC calculates directional centrality from the local covariance eigenvalues and their entropy. The implementation averages the results for neighborhood sizes 10 and 15 and uses the dataset-specific DCC weight.

FF encodes the original PCA branch and the SISD and DCC enhanced branch with separate heterogeneous graph transformer encoders. The resulting representations are concatenated before classification.

The optimizer is AdamW with learning rate 1.5e-3, learning-rate decay 0.98 per epoch, weight decay 1e-3, and gradient clipping at 5.0. Negative sampling uses a negative-to-positive ratio of 1.0.

## Ablation

The ablation program retrains only three variants: w/o DCC, w/o FF, and w/o SISD. The Full DGCGT rows are imported from the matching split-level benchmark CSV. Mouse_retina uses 30 ablation epochs. B_lymphoma and PBMCs_sampled use 40 ablation epochs.

## Optional R baselines

run_r_baselines.R evaluates FiRE, GapClust, and RaceID on an expression matrix and binary labels. R and the required packages are optional. See R_baselines_README.md for the command and method-specific requirements.
