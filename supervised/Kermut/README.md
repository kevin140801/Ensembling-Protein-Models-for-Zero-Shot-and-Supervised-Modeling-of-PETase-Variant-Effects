# Kermut

## Overview

This directory contains the notebooks used for supervised mutation effect prediction with Kermut.

Kermut combines multiple sources of information, including:
- ESM-2 protein embeddings
- ESM-2 zero-shot mutation scores
- Protein structure information
- ProteinMPNN features

The model is trained on experimentally measured mutation effects and evaluated using dataset-specific train/validation/test splits.

## Notebooks

### `Kermut_supervised.ipynb`

Main notebook for training and evaluating Kermut on the benchmark datasets.

The notebook includes:
- Data preparation and preprocessing
- Loading of embeddings, structures and zero-shot features
- Gaussian Process model training
- Prediction on validation and test sets
- Evaluation using Spearman correlation, NDCG@10% and R²
- Experiments on both single-mutant and multi-mutant datasets
- Generation of prediction files used in the ensemble analyses

## Notes

Kermut was evaluated using the same benchmark datasets and data splits as the other supervised models to enable direct performance comparisons.

Different input configurations were explored during the project, including modified Kermut variants using alternative embeddings and ensemble-derived zero-shot features. The final notebook contains the workflow used to generate the results reported in the thesis.

