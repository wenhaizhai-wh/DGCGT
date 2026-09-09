R baseline description

The file run_r_baselines.R evaluates FiRE, GapClust, and RaceID on an expression matrix and binary labels.

The program writes F1, Precision, Recall, AUPRC, ROC AUC, cell level scores, and an error record for methods that are not available in the R environment.

Run the following command from the codes/DGCGT folder.

Rscript run_r_baselines.R expression.csv labels.csv results.csv scores.csv 50 5 5
