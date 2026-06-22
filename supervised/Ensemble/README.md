# Supervised Ensemble

## Overview

This directory contains the notebook used to construct and evaluate supervised model ensembles.

The ensemble combines predictions from the supervised models used in this project:
- Fine-tuned ESM-2
- FoldVision
- Kermut

The main goal is to evaluate whether combining complementary supervised model predictions improves mutation effect prediction compared to individual models.

## Notebook

### `supervised_ensemble.ipynb`

Builds and evaluates weighted rank-based ensembles of supervised model predictions.

The notebook includes:
- Loading test-set predictions from fine-tuned ESM-2, FoldVision and Kermut
- Conversion of model predictions into percentile ranks
- Grid-search optimization of ensemble weights on selected reference datasets
- Evaluation of individual models and weighted ensembles using Spearman correlation and NDCG@10%
- Generation of the final reference-dataset performance plot
- Blind-style evaluation on the DLG4 abundance and binding datasets
- Evaluation of DLG4 performance using Spearman correlation, NDCG@10% and R²
- Generation of the final DLG4 blind-style performance plot

## Notes

The ensemble is rank-based to make predictions from different supervised models comparable despite different score scales.

Two supervised ensemble variants are evaluated:
- ESM-2 + FoldVision weighted rank ensemble
- ESM-2 + FoldVision + Kermut weighted rank ensemble

For the DLG4 blind-style evaluation, additional Kermut variants are included to compare the original Kermut model with a modified Kermut setup using fine-tuned ESM-2 embeddings and zero-shot ensemble features.
