"""Smoke tests for unphold.visualize on synthetic data, no phonopy I/O."""

import matplotlib

matplotlib.use("Agg")

import numpy
import pytest
from ase.atoms import Atoms as aseAtoms
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


def _make_bilayer_atoms() -> aseAtoms:
    cell = numpy.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 20.0]])
    positions = numpy.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 1.0, 5.0],
            [0.0, 0.0, 8.0],
            [1.0, 1.0, 8.0],
        ]
    )
    return aseAtoms(symbols=["C", "C", "C", "C"], cell=cell, positions=positions, pbc=True)


def test_import():
    from unphold.visualize import plot_layer_mode_2d, visualize_BZ_2d, visualize_cell_2d, visualize_kpath_2d

    assert callable(visualize_cell_2d)
    assert callable(visualize_BZ_2d)
    assert callable(visualize_kpath_2d)
    assert callable(plot_layer_mode_2d)


def test_visualize_cell_2d_default_ax():
    from unphold.visualize import visualize_cell_2d

    atoms = _make_bilayer_atoms()
    ax = visualize_cell_2d(atoms)
    assert isinstance(ax, Axes)


def test_visualize_cell_2d_given_ax():
    import matplotlib.pyplot as plt

    from unphold.visualize import visualize_cell_2d

    atoms = _make_bilayer_atoms()
    fig, ax = plt.subplots()
    returned_ax = visualize_cell_2d(atoms, ax=ax)
    assert returned_ax is ax
    plt.close(fig)


def test_visualize_BZ_2d_default_and_repeat():
    import matplotlib.pyplot as plt

    from unphold.visualize import visualize_BZ_2d

    atoms = _make_bilayer_atoms()
    fig, ax = plt.subplots()
    returned_ax = visualize_BZ_2d(atoms, ax=ax)
    assert returned_ax is ax

    fig2, ax2 = plt.subplots()
    returned_ax2 = visualize_BZ_2d(atoms, ax=ax2, repeat=(1, 1))
    assert returned_ax2 is ax2
    plt.close(fig)
    plt.close(fig2)


def test_visualize_kpath_2d_requires_coordinates():
    from unphold.visualize import visualize_kpath_2d

    atoms = _make_bilayer_atoms()
    with pytest.raises(ValueError):
        visualize_kpath_2d(atoms)


def test_visualize_kpath_2d_fractional_and_cartesian():
    import matplotlib.pyplot as plt

    from unphold.visualize import visualize_kpath_2d

    atoms = _make_bilayer_atoms()
    frac = numpy.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cart = numpy.array([[0.1, 0.1, 0.0]])

    fig, ax = plt.subplots()
    visualize_kpath_2d(atoms, fractional_coordinate=frac, ax=ax)
    visualize_kpath_2d(atoms, cartesian_coordinate=cart, ax=ax)
    visualize_kpath_2d(atoms, fractional_coordinate=frac, cartesian_coordinate=cart, ax=ax)
    plt.close(fig)


def test_plot_layer_mode_2d_requires_axes():
    from unphold.visualize import plot_layer_mode_2d

    atoms = _make_bilayer_atoms()
    disp = numpy.zeros((len(atoms), 3))
    with pytest.raises(ValueError):
        plot_layer_mode_2d(atoms, disp)


def test_plot_layer_mode_2d_single_mode():
    import matplotlib.pyplot as plt

    from unphold.visualize import plot_layer_mode_2d

    atoms = _make_bilayer_atoms()
    rng = numpy.random.default_rng(0)
    disp = rng.normal(size=(len(atoms), 3))

    fig, ax = plt.subplots()
    mappables = plot_layer_mode_2d(atoms, disp, freqs=2.3, axes=ax)
    assert isinstance(fig, Figure)
    assert len(mappables) == 1
    plt.close(fig)


def test_plot_layer_mode_2d_multi_mode():
    import matplotlib.pyplot as plt

    from unphold.visualize import plot_layer_mode_2d

    atoms = _make_bilayer_atoms()
    rng = numpy.random.default_rng(0)
    disp = rng.normal(size=(2, len(atoms), 3))

    fig, axes = plt.subplots(1, 2)
    mappables = plot_layer_mode_2d(atoms, disp, freqs=[2.3, 2.4], axes=list(axes), add_colorbar=False)
    assert len(mappables) == 2
    plt.close(fig)


def test_plot_layer_mode_2d_axes_length_mismatch():
    import matplotlib.pyplot as plt

    from unphold.visualize import plot_layer_mode_2d

    atoms = _make_bilayer_atoms()
    rng = numpy.random.default_rng(0)
    disp = rng.normal(size=(2, len(atoms), 3))

    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_layer_mode_2d(atoms, disp, axes=ax)
    plt.close(fig)


def test_plot_layer_mode_2d_shared_norm_not_rescaled():
    import matplotlib.pyplot as plt

    from unphold.visualize import plot_layer_mode_2d

    atoms = _make_bilayer_atoms()
    disp = numpy.zeros((len(atoms), 3))
    disp[:, 2] = 1.0  # uniform out-of-plane displacement

    shared_norm = Normalize(vmin=-5.0, vmax=5.0)
    fig, ax = plt.subplots()
    plot_layer_mode_2d(atoms, disp, axes=ax, norm=shared_norm, add_colorbar=False)
    assert ax.collections[0].norm.vmin == -5.0
    assert ax.collections[0].norm.vmax == 5.0
    plt.close(fig)
