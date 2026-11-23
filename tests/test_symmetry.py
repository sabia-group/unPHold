"""Regression checks for the active rotation and source-to-target mapping."""

import numpy as np
from ase import Atoms

from unphold.symmetry import _rotate_fc, map_atoms_c3, rotmat_xy


def test_rotate_xz_block_counterclockwise():
    """Check the destination atom pair and Cartesian signs for one active rotation.

    Viewed from +z, a +120 degree rotation cycles the triangle 0 -> 1 -> 2 -> 0,
    so the source block (0, 1) moves to (1, 2). Its only component is xz:
    the displacement axis z stays fixed while the force axis x rotates to
    (-1/2, sqrt(3)/2, 0). This gives independent expected xz and yz entries.
    """
    cell = [[6, 0, 0], [3, 3 * np.sqrt(3), 0], [0, 0, 12]]
    positions = [[1, 0, 0], [-0.5, np.sqrt(3)/2, 0], [-0.5, -np.sqrt(3)/2, 0]]
    atoms = Atoms("C3", positions=positions, cell=cell, pbc=True)
    perm, _, _ = map_atoms_c3(atoms)
    np.testing.assert_array_equal(perm, [1, 2, 0])

    fc = np.zeros((3, 3, 3, 3))
    fc[0, 1, 0, 2] = 1
    expected = np.zeros_like(fc)
    expected[1, 2, 0, 2] = -0.5
    expected[1, 2, 1, 2] = np.sqrt(3)/2
    np.testing.assert_allclose(_rotate_fc(fc, perm, rotmat_xy(2*np.pi/3)), expected, atol=1e-14)
