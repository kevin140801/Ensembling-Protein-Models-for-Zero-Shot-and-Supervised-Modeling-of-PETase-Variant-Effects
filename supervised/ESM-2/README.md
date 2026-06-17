# ESM-2 Fine-Tuning

## Overview

This directory contains the code used for supervised mutation effect prediction with a fine-tuned ESM-2 model.

The workflow consists of three main steps:
1. Preparing dataset-specific train, validation and test splits.
2. Fine-tuning ESM-2 with a task-specific feed-forward prediction head.
3. Evaluating the trained models on held-out test sets.

The supervised model uses a pretrained ESM-2 encoder and extracts the sequence representation from the ESM-2 classification token (position 0 embedding). This representation is passed to a feed-forward prediction head consisting of batch normalization, ReLU activation, dropout and a final prediction layer.

## Files

### `data_preprocess.ipynb`

Prepares the benchmark datasets for ESM-2 fine-tuning.

The notebook:
- Loads the predefined train, validation and test splits.
- Standardizes sequence and mutation information.
- Creates dictionaries containing sequences and labels.
- Exports the files required for model training and evaluation.

### `training.py`

Main training script for supervised ESM-2 fine-tuning.

The script:
- Loads pretrained ESM-2 weights.
- Adds a task-specific feed-forward prediction head.
- Trains the model using the prepared benchmark datasets.
- Supports distributed multi-GPU training.
- Saves model checkpoints and training logs.

### `evaluation.py`

Evaluation script for trained ESM-2 models.

The script:
- Loads trained model checkpoints.
- Generates predictions for validation or test datasets.
- Exports prediction files used in downstream analyses.

### `util_model.py`

Defines the supervised ESM-2 architecture.

The model consists of:
- A pretrained ESM-2 encoder.
- A feed-forward prediction head with batch normalization, ReLU activation and dropout.

### `util_data.py`

Dataset and dataloader utilities.

Contains dataset classes used for:
- Sequence-only prediction tasks.
- Sequence and small-molecule input tasks.

### `util_helper.py`

Helper functions used throughout training and evaluation.
