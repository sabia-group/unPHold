# unPHold

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17714100.svg)](https://doi.org/10.5281/zenodo.17714100)
[![Documentation Status](https://readthedocs.org/projects/unphold/badge/?version=latest)](https://unphold.readthedocs.io/en/latest/?badge=latest)

Phonon band unfolding from supercells to primitive cells, with mode-character metrics.
Designed for moiré and twisted bilayer systems; works for any supercell geometry.

**Documentation**: [unphold.readthedocs.io](https://unphold.readthedocs.io/en/latest/)

## Installation

```bash
pip install .
```

Or in editable mode for development:

```bash
pip install -e ".[dev]"
```

**Dependencies**: Python >= 3.10, NumPy, ASE, Phonopy >= 3.0, tqdm.
