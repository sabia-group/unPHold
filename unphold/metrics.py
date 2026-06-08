"""Phonon mode character metrics.

Functions for quantifying the physical character of phonon modes:
acoustic participation ratio (APR), longitudinality (L), verticality (V),
and Gaussian band expansion for plotting.
"""

from typing import Union

import numpy
from ase.atoms import Atoms as aseAtoms
from phonopy import Phonopy

from .atoms import atoms_ph2ase


def band_expansion():
    pass
