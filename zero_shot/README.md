# Zero-Shot Mutation Effect Prediction

This directory contains all notebooks used for the zero-shot mutation effect prediction analysis.

The evaluated models include:
- ESM-2
- ProSST
- VESPA
- Rosetta
- Rank-based ensemble

The goal of the zero-shot analysis is to assess whether pretrained protein language models, evolutionary predictors and structure-based methods can prioritize mutation effects without task-specific training data. Model performance is evaluated on multiple deep mutational scanning (DMS) datasets using Spearman rank correlation and NDCG@10%.
Predictions for EVE, ESM-3 and ESM-IF were externally provided by my supervisor and are therefore not recomputed in this repository. The corresponding prediction files are stored in the `predictions/` directory under the respective model subfolders.

In addition to the benchmark evaluation, this directory contains notebooks for:
- Multi-mutant variant analysis
- Rank-based ensemble construction and optimization
- Blind evaluation on DLG4 datasets
- PETase Tournament submission generation

Each model-specific subdirectory contains the corresponding notebooks and additional details about the implemented workflow.
