# ESM-2 Zero-Shot

## Notebooks

### `ESM2_zero-shot.ipynb`

Computes ESM-2 zero-shot predictions for the single-mutant benchmark datasets.  
Mutation effects are scored using log-likelihood differences between wild-type and mutant amino acids and evaluated with Spearman correlation and NDCG@10%.

### `ESM2_zero-shot_multi.ipynb`

Applies the ESM-2 zero-shot approach to multi-mutant variants.  
Scores are computed additively by summing the position-wise log-likelihood differences over all mutated positions.
