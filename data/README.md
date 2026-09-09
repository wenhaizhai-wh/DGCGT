Data Description

This folder explains the data format required to run DGCGT.

The repository does not include the original single-cell data. The datasets are large and may have usage restrictions, so they are not uploaded to GitHub. To run an experiment on real data, prepare the required files separately and provide the data folder path to the program.

RNA data should include the following files.

Gene_Cell.mtx is the gene by cell expression matrix.

Cell_names.tsv contains one cell name per line.

Gene_names.tsv contains one gene name per line.

Cell_type.tsv contains one cell-type label per line.

For paired RNA and ATAC data, the following files are also required.

Peak_Cell.mtx is the peak by cell ATAC matrix.

Peak_names.tsv contains the peak names.

Gene_Peak.mtx contains the connections between genes and peaks.

If the rare cell types need to be specified explicitly, add a file named Rare_types.tsv and write one rare cell type on each line. If this file is not provided, the program uses the two least frequent cell types as the rare-cell class.

The cell order must be consistent across all files. Gene names, peak names, and cell-type labels must match the corresponding matrix dimensions.

The simulated data do not require a download. The program can generate simulated data automatically for testing DGCGT. The relevant script is located at codes/DGCGT/run_simulated.py.
