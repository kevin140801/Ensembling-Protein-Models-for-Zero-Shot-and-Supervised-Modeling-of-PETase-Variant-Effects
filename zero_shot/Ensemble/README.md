# Zero-Shot Ensemble and PETase Submission

## Notebooks

### `zero_shot_ensemble.ipynb`

Builds the final rank-based zero-shot ensemble from the individual model predictions.  
Model scores are converted into ranks to make outputs from different predictors comparable and ensemble weights are optimized on the benchmark datasets. The final ensemble is evaluated using Spearman correlation and NDCG@10%, including the blind DLG4 datasets.

### `PETase_zero_shot_submission_ensemble.ipynb`

Applies the optimized zero-shot ensemble to the PETase Tournament prediction files.  
The notebook generates the final PETase submission by combining ranked model predictions and mapping the resulting ensemble scores to the required submission format.

It also includes an optional comparison of different PETase submission files.
