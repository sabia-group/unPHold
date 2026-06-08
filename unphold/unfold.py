"""Core phonon unfolding classes.

Provides:
- :class:`Unfold`: general supercell → unitcell phonon band unfolding
- :class:`UnfoldTwistBilayer`: convenience wrapper for twisted bilayer systems (stub)
- :func:`concatenate_bands`: helper to merge k-path segments from Phonopy
"""

import os
import pickle
import time
from typing import Literal, Optional, Union

import numpy
from ase.atoms import Atoms as aseAtoms
from ase.build.supercells import make_supercell
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.band_structure import BandStructure
from phonopy.units import VaspToCm, VaspToEv, VaspToTHz
from tqdm import tqdm

from .atoms import match_two_atoms
from .metrics import band_expansion
