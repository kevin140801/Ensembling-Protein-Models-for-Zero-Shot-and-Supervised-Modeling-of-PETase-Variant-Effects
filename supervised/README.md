# Supervised Mutation Effect Prediction

This directory contains all code used for the supervised mutation effect prediction analysis.

The supervised models are trained on experimentally measured mutation-effect data and evaluated on predefined train, validation and test splits. The goal is to assess whether task-specific training improves mutation effect prediction and to compare different model architectures and input representations.

The supervised models evaluated in this project include:
- Fine-tuned ESM-2
- Kermut
- FoldVision
- Supervised ensembles

Model performance is evaluated using:
- Spearman rank correlation
- NDCG@10%
- Coefficient of determination (R²)

The benchmark datasets include multiple proteins and assays, covering both single-mutant and multi-mutant prediction tasks.

## Directory Structure

### `ESM2/`

Contains the supervised ESM-2 fine-tuning workflow, including data preparation, model training and evaluation.

### `Kermut/`

Contains the Kermut training and evaluation pipeline.

### `FoldVision/`

Contains the FoldVision structure-based prediction workflow.
Mutant PDB structures are generated using the Rosetta workflow implemented in:

`zero_shot/Rosetta/Rosetta_zero-shot.ipynb`

### `Ensemble/`

Contains notebooks used to combine predictions from multiple supervised models.
The ensemble analysis evaluates whether complementary information from different supervised predictors can improve mutation effect prediction performance.

## Notes

All supervised models were evaluated using identical train, validation and test splits to ensure fair comparisons across methods.
The resulting prediction files were subsequently used for model comparison, blind DLG4 evaluation, multi-mutant analyses and ensemble construction.
