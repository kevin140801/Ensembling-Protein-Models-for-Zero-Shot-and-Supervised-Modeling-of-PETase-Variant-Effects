# FoldVision

## Overview

This directory contains the code used for supervised mutation effect prediction with FoldVision.

FoldVision uses protein structure information as input. For each variant, mutant PDB structures are converted into FoldVision-compatible 3D point representations and used to train supervised prediction models on the datasets.

The workflow consists of four main steps:

1. Preparing dataset-specific train, validation and test CSV files.
2. Generating mutant PDB structures for all variants.
3. Preprocessing PDB files into FoldVision-compatible 3D point representations.
4. Training and evaluating FoldVision models on the prepared datasets.

## Files

### `data_preprocess.ipynb`

Prepares the benchmark datasets for FoldVision.

The notebook:
- Loads the predefined train, validation and test splits.
- Converts variant identifiers into FoldVision-compatible protein IDs.
- Creates CSV files containing protein identifiers and labels.
- Checks that train, validation and test splits are consistent and non-overlapping.

### `preprocess_pdb_dir.py`

Preprocesses directories of mutant PDB structures for FoldVision.

The script:
- Reads all PDB files in a given directory.
- Converts protein structures into 3D point-list representations.
- Saves processed `.npz` files.
- Creates the required `bounding_boxes.npy` file.

### `train.py`

Main training script for supervised FoldVision prediction.

The script:
- Loads preprocessed FoldVision structure inputs.
- Initializes the FoldVision encoder and prediction head.
- Trains the model on the training split.
- Uses the validation split for model selection.
- Saves checkpoints and training outputs.

### `evaluate.py`

Evaluation script for trained FoldVision models.

The script:
- Loads a trained FoldVision checkpoint.
- Generates predictions on validation or test data.
- Supports multiple augmented prediction runs.
- Exports prediction files and evaluation metrics.

### `embed_proteins.py`

Generates FoldVision encoder embeddings for preprocessed protein structures.

The script:
- Loads preprocessed FoldVision inputs.
- Runs the FoldVision encoder.
- Saves averaged embeddings across multiple augmented runs.

## Notes

Mutant PDB structures are required before running `preprocess_pdb_dir.py`.
In this project, mutant structures were generated using the Rosetta-based workflow implemented in:

`zero_shot/Rosetta/Rosetta_zero-shot.ipynb`

Starting from experimentally determined wild-type structures and variant definitions, Rosetta was used to generate mutant PDB files, which were subsequently processed for FoldVision.
