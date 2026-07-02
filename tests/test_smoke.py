"""Smoke tests — verify the package is importable and core classes instantiate."""

import numpy
import pytest


def test_import():
    import unphold  # noqa: F401


def test_public_api():
    from unphold import Unfold, UnfoldTwistBilayer
    from unphold.utils import concatenate_bands
    from unphold.metrics import compute_APR, compute_L, compute_V, compute_V_p2
    from unphold.utils import band_expansion, gaussian_function

    assert callable(Unfold)
    assert callable(concatenate_bands)
    assert callable(compute_APR)


def test_gaussian_function_scalar_mu():
    from unphold.utils import gaussian_function

    x = numpy.linspace(-3, 3, 100)
    g = gaussian_function(x, mu=0, sigma=1.0)
    assert g.shape == x.shape
    # integral ≈ 1
    assert abs(numpy.trapezoid(g, x) - 1.0) < 0.01


def test_gaussian_function_array_mu():
    from unphold.utils import gaussian_function

    x = numpy.linspace(-5, 5, 200)
    mu = numpy.array([0.0, 1.0, 2.0])
    g = gaussian_function(x, mu=mu, sigma=0.5)
    assert g.shape == (3, 200)


def test_band_expansion_shape():
    from unphold.utils import band_expansion

    energies = numpy.array([10.0, 20.0, 30.0])
    grid = numpy.linspace(0, 40, 200)
    result = band_expansion(energies, grid, sigma=1.0)
    assert result.shape == (3, 200)


def test_concatenate_bands_simple():
    from unphold.utils import concatenate_bands

    seg0 = numpy.linspace([0, 0, 0], [0.5, 0, 0], 5)  # shape (5, 3)
    seg1 = numpy.linspace([0.5, 0, 0], [1.0, 0, 0], 5)
    kpts, indices = concatenate_bands([seg0, seg1], connections=[True, False])
    # connected: seg0 loses last point → 4 + 5 = 9
    assert kpts.shape[0] == 9
    assert indices[0] == 0


def test_fractional_part_around_zero():
    from unphold.utils import fractional_part_around_zero

    arr = numpy.array([-1.5, -0.5, 0.0, 0.25, 0.5, 1.5, 2.5])
    result = fractional_part_around_zero(arr)
    assert result.shape == arr.shape
    assert numpy.all(result >= -0.5) and numpy.all(result < 0.5)
    # non-halfway values are unaffected by the rounding convention
    assert numpy.isclose(fractional_part_around_zero(numpy.array([0.25]))[0], 0.25)


def test_match_two_2d_atoms_pbc_with_2d_frac_shift_identity():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import match_two_2d_atoms_pbc_with_2d_frac_shift

    cell = numpy.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 20.0]])
    positions = numpy.array([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]])
    a = aseAtoms(symbols=["C", "C"], cell=cell, positions=positions, pbc=True)
    b = a.copy()

    result = match_two_2d_atoms_pbc_with_2d_frac_shift(a, b, shift_0_seg=3, shift_1_seg=3)
    assert "atoms_indices_a2b" in result
    numpy.testing.assert_array_equal(result["atoms_indices_a2b"], numpy.array([0, 1]))
    numpy.testing.assert_array_equal(result["atoms_indices_b2a"], numpy.array([0, 1]))


def test_match_two_2d_atoms_pbc_with_2d_frac_shift_no_match():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import match_two_2d_atoms_pbc_with_2d_frac_shift

    cell = numpy.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 20.0]])
    a = aseAtoms(symbols=["C", "C"], cell=cell, positions=[[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]], pbc=True)
    b = aseAtoms(symbols=["C", "C"], cell=cell, positions=[[0.3, 0.7, 5.0], [1.6, 0.2, 5.0]], pbc=True)

    result = match_two_2d_atoms_pbc_with_2d_frac_shift(
        a, b, shift_0_frac=0.01, shift_0_seg=3, shift_1_frac=0.01, shift_1_seg=3
    )
    assert "atoms_indices_a2b" not in result
    assert "atoms_dist_list" in result


def test_relax_by_spring_origin_spring_pulls_atom_back():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import RelaxBySpring

    cell = numpy.eye(3) * 10.0
    positions = numpy.array([[5.0, 5.0, 5.0]])
    atoms = aseAtoms(symbols=["C"], cell=cell, positions=positions, pbc=True)

    relax = RelaxBySpring(atoms)
    relax.install_origin_spring(k=1.0)
    displaced = relax.positions.copy()
    displaced[0] += 1.0
    relax._pos = displaced  # simulate a displacement away from the origin spring target

    relax.relax(steps=200, delta=0.05)
    assert numpy.allclose(relax.positions, positions, atol=1e-2)
    info = relax.get_spring_info()
    assert info == {"origin_springs": 1, "layer_springs": 0, "neighbor_springs": 0}

    relax.clear_springs()
    assert relax.get_spring_info() == {"origin_springs": 0, "layer_springs": 0, "neighbor_springs": 0}


# TODO: add Unfold integration test with a minimal graphene 2x2 supercell
# Requires phonopy force constants — either fixture files or a synthetic dynamical matrix
