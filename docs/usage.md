# Usage

!!! note
    Full usage examples will be added in Stage 2 alongside integration tests.

## Basic workflow

1. Create `phonopy.yaml` with finite displacement structures.
1. Calculate forces with DFT or MLIP.
1. Load forces and compute supercell modes with Phonopy.
1. Initialize k-point path and calculate unfolding weights.
1. Visualize unfolded band structure and realspace motion.

## Unfolding weight

The unfolding weight of supercell band \(n\) onto primitive-cell k-point \(\mathbf{k}\) is:

\[
w_{\mathbf{k},n} = \frac{1}{N_\text{uc}} \sum_i \left| \langle \phi^{\text{uc}}_{\mathbf{k},i} | \Psi^{\text{sc}}_{\mathbf{k},n} \rangle \right|^2
\]

where \(N_\text{uc}\) is the number of primitive cells in the supercell.

## Mode character metrics

```python
apr = unphold.metrics.compute_APR(...)
l   = unphold.metrics.compute_L(...)
v   = unphold.metrics.compute_V(...)
```
