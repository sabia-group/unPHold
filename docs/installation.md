# Installation

## Requirements

- Python ≥ 3.10
- [NumPy](https://numpy.org/) ≥ 1.24
- [ASE](https://wiki.fysik.dtu.dk/ase/) ≥ 3.23
- [Phonopy](https://phonopy.github.io/phonopy/) ≥ 3.0
- [tqdm](https://tqdm.github.io/) ≥ 4.0

## Install with pip

To install from GitHub:
```bash
pip install git+https://github.com/sabia-group/unPHold.git
```
<!-- TODO: update this link to a stable version -->

To install from source:
```bash
git clone https://github.com/sabia-group/unPHold
cd unPHold
pip install -e .
```

To install from source with extras:
```bash
pip install -e ".[dev]"   # testing: pytest, pytest-cov, ruff
pip install -e ".[docs]"  # documentation: mkdocs, mkdocstrings
pip install -e ".[all]"   # both
```

## Conda environment setup

```bash
env_name="unphold"
conda create -n ${env_name} python=3.12 -y
conda install -n ${env_name} -c conda-forge "numpy>=1.24" "ase>=3.23" "phonopy>=3.0" "tqdm>=4.0" pip -y
# then follow above pip install instructions, e.g.
conda run -n ${env_name} pip install git+https://github.com/sabia-group/unPHold.git
```

