DGCGT parameter guide

This file records the main settings used by the supplied experiments. Feature selection, scaling, and PCA are fitted on the training part of each split.

Mouse retina uses hidden dimension 128, one graph layer, four attention heads, dropout 0.30, DCC weight 3.0, DCC neighborhood sizes 10 and 15, SISD alpha 0.10, one SISD step, 13 ensemble members, and 80 training epochs.

B cell lymphoma uses hidden dimension 128, one graph layer, four attention heads, dropout 0.10, DCC weight 2.0, DCC neighborhood sizes 10 and 15, SISD alpha 0.15, one SISD step, 13 ensemble members, and 40 training epochs.

Sampled PBMC data use hidden dimension 128, one graph layer, four attention heads, dropout 0.20, DCC weight 3.0, DCC neighborhood sizes 10 and 15, SISD alpha 0.10, one SISD step, 13 ensemble members, and 40 training epochs.

The simulated data use hidden dimension 128, one graph layer, four attention heads, dropout 0.20, DCC weight 0.8, DCC neighborhood sizes 10 and 15, SISD alpha 0.40, three SISD steps, five ensemble members, and 50 training epochs.

The formal real data experiments use five repeated stratified splits. Each split uses 80 percent of the cells for training and 20 percent for testing. The random seed is 2026. The fixed decision threshold is 0.5.

Preprocessing

Mouse retina and sampled PBMC data use 150 highly variable genes, 100 peaks, and RNA and ATAC PCA dimensions of 40 and 10. B cell lymphoma uses 500 genes, 300 peaks, and PCA dimensions of 25 and 25. Simulated data use 200 genes, 100 peaks, and PCA dimensions of 25 and 25.

The cell graph uses 15 nearest neighbors. The heterogeneous graph uses 12 cell to gene neighbors, 10 cell to peak neighbors, and 3 gene to peak neighbors.

Count like data are library size normalized and log transformed. Feature selection and PCA are fitted separately inside each training split. The simulated data are continuous noisy values and are not log transformed.

Model components

SISD applies multi scale graph smoothing. Its neighborhood size, alpha, and number of steps are selected from the rare cell fraction in the training split.

DCC calculates directional centrality from local covariance eigenvalues and entropy. The implementation uses neighborhood sizes 10 and 15.

FF encodes the original and enhanced feature branches separately and combines their representations before classification.

The optimizer is AdamW with learning rate decay, weight decay, and gradient clipping. Early stopping is not used in the supplied protocols.

Ablation

The ablation command retrains the variants without DCC, without FF, and without SISD. The Full DGCGT values are read from the matching split level benchmark file.

R baselines

The optional R program is run_r_baselines.R. It evaluates FiRE, GapClust, and RaceID when the required R packages are installed. The output uses F1, Precision, Recall, AUPRC, and ROC AUC.
