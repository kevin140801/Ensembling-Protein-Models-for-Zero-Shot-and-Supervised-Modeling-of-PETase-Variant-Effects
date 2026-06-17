# VESPA Zero-Shot

## Notebooks

### `VESPA_zero-shot.ipynb`

Computes zero-shot mutation effect predictions using VESPA for the benchmark datasets.

VESPA predicts the impact of amino acid substitutions from evolutionary sequence information without requiring task-specific training data. The notebook processes VESPA predictions for the benchmark datasets, computes mutation effect scores, and evaluates performance using Spearman correlation and NDCG@10%.

For multi-mutant variants, additive scores are generated in the ensemble workflow by summing the corresponding single-mutant VESPA predictions.
