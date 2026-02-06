# Checkpoints

This directory contains pre-trained and fine-tuned model checkpoints for Proteoformer.

## Pre-trained Model

### proteoformer-base

The base Proteoformer model pre-trained on large-scale protein sequences and proteoform corpus. This checkpoint (checkpoint-197000) serves as the foundation for all downstream fine-tuning tasks.


## Fine-tuned Checkpoints

### PTM_bench

Fine-tuned models for PTM site prediction tasks on various modification types:

| Checkpoint | Modification Type | Target Residue |
|------------|-------------------|----------------|
| `acetylation_k_checkpoint_proteoformer.pt` | Acetylation | Lysine (K) |
| `glycosylation_n_checkpoint_proteoformer.pt` | Glycosylation | Asparagine (N) |
| `methylation_k_checkpoint_proteoformer.pt` | Methylation | Lysine (K) |
| `methylation_r_checkpoint_proteoformer.pt` | Methylation | Arginine (R) |
| `phosphorylation_st_checkpoint_proteoformer.pt` | Phosphorylation | Serine/Threonine (S/T) |
| `phosphorylation_y_checkpoint_proteoformer.pt` | Phosphorylation | Tyrosine (Y) |
| `sumoylation_k_checkpoint_proteoformer.pt` | SUMOylation | Lysine (K) |
| `ubiquitination_k_checkpoint_proteoformer.pt` | Ubiquitination | Lysine (K) |

### PTMfunctionalassociation

Fine-tuned model for predicting functional associations of PTM sites:

| Checkpoint | Task |
|------------|------|
| `PTM_sites_functional_association_prediction_ckpt.pt` | PTM Functional Association Prediction |

### Variant_PTM_effect

Fine-tuned models for predicting the effects of genetic variants on PTM sites:

| Checkpoint | Task |
|------------|------|
| `Direct_Variant_PTM_effect_ckpt.pt` | Direct variant effect on PTM sites |
| `Indirect_Variant_PTM_effect_ckpt.pt` | Indirect variant effect on PTM sites |

## Usage

For detailed usage examples, please refer to the [examples](../examples) directory.

## Download

Pre-trained and fine-tuned checkpoints will be coming soon.

## License

These checkpoints are released under the Apache 2.0 License.
