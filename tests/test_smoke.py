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


# TODO: add Unfold integration test with a minimal graphene 2x2 supercell
# Requires phonopy force constants — either fixture files or a synthetic dynamical matrix
