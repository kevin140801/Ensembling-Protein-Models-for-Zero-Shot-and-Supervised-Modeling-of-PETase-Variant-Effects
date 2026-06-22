# Ensembling Protein Models for Zero Shot and Supervised Modeling of PETase Variant Effects

Master Thesis

## Overview

This repository contains the code used in my Master's thesis on mutation effect prediction. The project investigates how different protein foundation models can be combined to improve the prediction of variant effects in both zero-shot and supervised settings.

The work focuses on benchmark datasets from deep mutational scanning experiments and explores the complementarity of sequence-based, structure-based and evolutionary protein models through rank-based ensemble methods.

## Repository Structure

```text
.
├── data/
│   └── Datasets and links
├── zero_shot/
│   ├── ESM-2
│   ├── ProSST
│   ├── VESPA
│   ├── Rosetta
│   └── Ensemble
└── supervised/
    ├── ESM-2 Fine-Tuning
    ├── Kermut
    ├── FoldVision
    └── Ensemble
```

### Zero-Shot Models

The following zero-shot models are included in this repository:

- ESM-2
- ProSST
- VESPA
- Rosetta

### Supervised Models

The following supervised models are included in this repository:

- Fine-tuned ESM-2
- Kermut
- FoldVision

## Data Availability

The datasets used in this project are available on Zenodo: https://zenodo.org/records/20792349

Included datasets:

- UBE4B
- GRB2
- PTEN activity
- PTEN abundance
- Alpha-Amylase
- DLG4 abundance
- DLG4 binding
- PETase (zero-shot)

Original benchmark datasets were obtained from:

- https://github.com/gitter-lab/metl-pub
- https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023

## Notes

EVE, ESM-3 and ESM-IF were used in the analyses presented in the thesis. The corresponding code is not included in this repository because access to these models was provided by my supervisor.
