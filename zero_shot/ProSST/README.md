# ProSST Zero-Shot

## Notebooks

### `ProSST_zero-shot.ipynb`

Computes zero-shot mutation effect predictions using the pretrained ProSST model for the single-mutant benchmark datasets.  
Mutation effects are estimated from log-likelihood differences between wild-type and mutant amino acids conditioned on sequence-derived structural tokens and evaluated with Spearman correlation and NDCG@10%.

### `ProSST_zero-shot_multi.ipynb`

Applies the ProSST zero-shot scoring approach to multi-mutant variants.  
Scores are computed additively by summing the position-wise mutation scores across all mutated positions.
