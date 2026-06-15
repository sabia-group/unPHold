# Installation

## Requirements

- Python ≥ 3.10
- [NumPy](https://numpy.org/) ≥ 1.24
- [ASE](https://wiki.fysik.dtu.dk/ase/) ≥ 3.22
- [Phonopy](https://phonopy.github.io/phonopy/) ≥ 2.20
- [tqdm](https://tqdm.github.io/) ≥ 4.0

## Install from PyPI

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

To install with extras:
```bash
pip install -e ".[dev]"   # testing
pip install -e ".[docs]"  # documentation
```

## Conda environment setup

```bash
env_name="unphold"
conda create -n $env_name python=3.12 -y
conda install -n $env_name "numpy>=1.24" "ase>=3.22" "phonopy>=2.20" "tqdm>=4.0" pip -y
# install unPHold in the conda environment using pip, e.g. from GitHub
conda run -n $env_name pip install git+https://github.com/sabia-group/unPHold.git
```

