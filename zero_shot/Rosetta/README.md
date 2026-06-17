# Rosetta Zero-Shot

## Notebooks

### `Rosetta_zero-shot.ipynb`

Computes zero-shot mutation effect predictions using the Rosetta energy function for the benchmark datasets.

For each dataset, mutant protein structures are generated from the experimental wild-type structure using PyRosetta. After side-chain repacking and local energy minimization, mutation effects are estimated as the difference in Rosetta total energy between the mutant and wild-type structures (ΔΔG-like score).

The notebook includes:
- Generation of mutant PDB structures (`mut_pdbs`)
- Rosetta energy evaluation of wild-type and mutant structures
- Computation of mutation effect scores
- Evaluation using Spearman correlation and NDCG@10%
