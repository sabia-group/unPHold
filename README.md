# unPHold

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17714099.svg)](https://doi.org/10.5281/zenodo.17714099)
[![Documentation Status](https://readthedocs.org/projects/unphold/badge/?version=latest)](https://unphold.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/sabia-group/unPHold/actions/workflows/tests_main.yml/badge.svg)](https://github.com/sabia-group/unPHold/actions/workflows/tests_main.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**unPHold** unfolds phonon band structures from a supercell Brillouin zone onto a smaller unit cell Brillouin zone.

**Documentation**: [unphold.readthedocs.io](https://unphold.readthedocs.io/en/latest/)

## Features

- Handles 3D and 2D systems, moiré superlattices, defective structures, and relaxed or deregistered structures.
- Per-mode unfolding weights and unfolded band structure visualization.
- Mode-character metrics: acoustic participation ratio (APR), longitudinality (L), and out-of-plane verticality (V).
- Visualization of real-space phonon displacements.
- Built on [phonopy](https://phonopy.github.io/phonopy/): unPHold reuses phonopy's force constants and structural data, so it works with any force-constant source, including density-functional theory (DFT) codes and machine-learning interatomic potentials (MLIPs).

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/sabia-group/unPHold.git
```

See the [installation guide](https://unphold.readthedocs.io/en/latest/installation/) for details.

## Quickstart

```python
from unphold import Unfold

unfold = Unfold(
    unitcell=uc_atoms,           # primitive cell (ase.Atoms)
    supercell=sc_atoms,          # supercell (ase.Atoms)
    transformation_matrix=tmat,  # supercell transformation matrix
)
unfold.set_kpts_in_unitcell(kpts, format="fractional")
unfold.calculate_sc_phonon(dyn_sc=ph_sc.dynamical_matrix)
unfold.calculate_weights()
spectral, grid, _ = unfold.calculate_spectral_function_on_grid(grid=freq_grid, sigma=0.1)
```

Example scripts are available at [`examples/`](examples/), including bulk Si, twisted bilayer graphene, and graphene monolayer with a vacancy. Step-by-step walkthroughs are in the [tutorials](https://unphold.readthedocs.io/en/latest/tutorials/si/).

## Citation

If you use unPHold in your research, please cite this Zenodo record: [10.5281/zenodo.17714099](https://doi.org/10.5281/zenodo.17714099).

## License

unPHold is distributed under the [MIT license](LICENSE).
