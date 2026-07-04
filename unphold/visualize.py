"""2D real-space and reciprocal-space visualization helpers.

Functions for plotting a structure's real-space cell, its reciprocal-space
Brillouin zone and k-paths, and per-mode atomic displacement patterns
(in-plane arrows, out-of-plane color). All functions operate on plain ASE
``Atoms`` objects and plain arrays, they have no dependency on ``Unfold``.
"""

from itertools import product

import numpy
from ase.atoms import Atoms as aseAtoms
from ase.dft.bz import bz_vertices
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize


def visualize_cell_2d(
    geom: aseAtoms,
    alpha_min: float = 0.2,
    alpha_max: float = 1.0,
    base_size: float = 1e1,
    ax: Axes | None = None,
) -> Axes:
    """Visualize the 2D projection of a cell and its atoms (z coordinate as opacity).

    Args:
        geom (aseAtoms): Structure to plot.
        alpha_min (float): Opacity of the lowest atom (smallest z).
        alpha_max (float): Opacity of the highest atom (largest z).
        base_size (float): Base marker size, scaled by atomic number and atom count.
        ax (Axes | None): Axes to draw into. A new figure/axes is created if not given.

    Returns:
        Axes: The axes drawn into.
    """
    if ax is None:
        ax = plt.gca()

    symbs: list[str] = geom.get_chemical_symbols()
    natoms = len(symbs)
    atomz_arr: numpy.ndarray = geom.get_atomic_numbers()  # Z number of atoms
    coords: numpy.ndarray = geom.get_positions()  # shape (natoms, xyz)
    cell: numpy.ndarray = geom.get_cell()  # shape (lat_vec, xyz)
    zmax, zmin = coords[:, 2].max(), coords[:, 2].min()
    if abs(zmax - zmin) < 1e-3:
        zmin = -(zmax + 1)

    z_order = 2.1 + 0.1 * (coords[:, 2] - zmin) / (zmax - zmin)  # from 2.1 to 2.2
    alpha_arr = alpha_min + (alpha_max - alpha_min) * (coords[:, 2] - zmin) / (zmax - zmin)
    alpha_arr[alpha_arr > 1.0] = 1.0
    alpha_arr[alpha_arr < 0.0] = 0.0
    size_arr = base_size / natoms**0.5 * atomz_arr**0.5
    color_dict = {symb: f"C{idx}" for idx, symb in enumerate(set(symbs))}
    color_list = [color_dict[symb] for symb in symbs]

    # plot atoms
    for i in range(natoms):
        ax.plot(
            coords[i, 0],
            coords[i, 1],
            marker="o",
            ms=size_arr[i],
            color=color_list[i],
            alpha=alpha_arr[i],
            zorder=z_order[i],
        )

    # plot bounding box
    cell_2d = [cell[0, :2], cell[1, :2]]
    ax.plot([0, cell_2d[0][0]], [0, cell_2d[0][1]], "k--")
    ax.plot([0, cell_2d[1][0]], [0, cell_2d[1][1]], "k--")
    ax.plot([cell_2d[0][0], cell_2d[0][0] + cell_2d[1][0]], [cell_2d[0][1], cell_2d[0][1] + cell_2d[1][1]], "k--")
    ax.plot([cell_2d[1][0], cell_2d[0][0] + cell_2d[1][0]], [cell_2d[1][1], cell_2d[0][1] + cell_2d[1][1]], "k--")

    ax.set_aspect("equal")
    ax.grid(True)

    return ax


def visualize_BZ_2d(
    geom: aseAtoms,
    plt_kwargs: dict | None = None,
    ax: Axes | None = None,
    repeat: tuple[int, int] = (0, 0),
) -> Axes:
    """Plot the 2D Brillouin zone edges (reciprocal xy-plane) of a cell.

    Takes/returns an ``ax`` so multiple BZs (e.g. monolayer PC vs. TBG SC) can be
    overlaid on the same axes to compare orientation and size.

    ``repeat`` draws a repeated-zone scheme: BZ replicas shifted by integer
    multiples of the reciprocal lattice vectors, from ``-repeat`` to ``+repeat``
    (inclusive) along each in-plane direction, centered on the original
    (Gamma-centered) BZ. Useful for comparing a small SC BZ against a much
    larger PC BZ: e.g. ``repeat=(6, 6)`` tiles the SC BZ so its replicas'
    corners/edges can be checked against the PC BZ and its k-points.

    Args:
        geom (aseAtoms): Structure whose cell defines the reciprocal lattice.
        plt_kwargs (dict | None): Extra kwargs forwarded to ``ax.plot`` for each edge.
        ax (Axes | None): Axes to draw into. A new figure/axes is created if not given.
        repeat (tuple[int, int]): Number of BZ replicas to draw along each in-plane
            reciprocal lattice direction, on each side of the origin.

    Returns:
        Axes: The axes drawn into.
    """
    if ax is None:
        ax = plt.gca()
    plt_kwargs = {} if plt_kwargs is None else dict(plt_kwargs)
    plt_kwargs.setdefault("color", "k")
    plt_kwargs.setdefault("linestyle", "-")

    icell = geom.cell.reciprocal()  # rows = reciprocal lattice vectors (no 2*pi factor)
    bz_edges = bz_vertices(icell, dim=2)  # dim=2 flattens the BZ onto the xy-plane

    na, nb = repeat
    for ia, ib in product(range(-na, na + 1), range(-nb, nb + 1)):
        shift = ia * icell[0, :2] + ib * icell[1, :2]
        for points, _normal in bz_edges:
            xy = numpy.concatenate([points, points[:1]])  # close the polygon edge
            ax.plot(xy[:, 0] + shift[0], xy[:, 1] + shift[1], **plt_kwargs)

    ax.set_aspect("equal")

    return ax


def visualize_kpath_2d(
    geom: aseAtoms,
    fractional_coordinate: numpy.ndarray | None = None,
    cartesian_coordinate: numpy.ndarray | None = None,
    plt_kwargs: dict | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot k-points on the reciprocal xy-plane, in the same coordinates as [visualize_BZ_2d][unphold.visualize.visualize_BZ_2d].

    Provide at least one of:

    - ``fractional_coordinate``: k-points as fractional (reduced) reciprocal-lattice
      coordinates, shape ``(npoints, 2 or 3)``. Converted to Cartesian via ``geom``'s
      reciprocal cell.
    - ``cartesian_coordinate``: k-points already in Cartesian reciprocal coordinates,
      shape ``(npoints, 2 or 3)``.

    Both may be given at once (e.g. a PC path in fractional PC coordinates plus an
    already-transformed SC path in Cartesian coordinates); they are plotted together.

    Args:
        geom (aseAtoms): Structure whose cell defines the reciprocal lattice.
        fractional_coordinate (numpy.ndarray | None): K-points in fractional coordinates.
        cartesian_coordinate (numpy.ndarray | None): K-points in Cartesian coordinates.
        plt_kwargs (dict | None): Extra kwargs forwarded to ``ax.plot``.
        ax (Axes | None): Axes to draw into. A new figure/axes is created if not given.

    Returns:
        Axes: The axes drawn into.

    Raises:
        ValueError: If neither ``fractional_coordinate`` nor ``cartesian_coordinate`` is given.
    """
    if fractional_coordinate is None and cartesian_coordinate is None:
        raise ValueError("must provide at least one of `fractional_coordinate` or `cartesian_coordinate`")

    if ax is None:
        ax = plt.gca()
    plt_kwargs = {} if plt_kwargs is None else dict(plt_kwargs)
    plt_kwargs.setdefault("marker", ".")
    plt_kwargs.setdefault("linestyle", ":")
    plt_kwargs.setdefault("color", "r")

    xy_parts = []
    if fractional_coordinate is not None:
        frac = numpy.atleast_2d(numpy.asarray(fractional_coordinate, dtype=float))
        icell = geom.cell.reciprocal()  # rows = reciprocal lattice vectors (no 2*pi factor)
        cart = frac[:, :3] @ icell  # fractional -> Cartesian reciprocal coordinates
        xy_parts.append(cart[:, :2])
    if cartesian_coordinate is not None:
        cart = numpy.atleast_2d(numpy.asarray(cartesian_coordinate, dtype=float))
        xy_parts.append(cart[:, :2])

    xy = numpy.concatenate(xy_parts, axis=0)
    ax.plot(xy[:, 0], xy[:, 1], **plt_kwargs)

    ax.set_aspect("equal")

    return ax


def plot_layer_mode_2d(
    atoms: aseAtoms,
    displacements: numpy.ndarray,
    freqs: float | numpy.ndarray | None = None,
    axes: Axes | list[Axes] = None,
    cmap: str = "bwr",
    norm: Normalize | None = None,
    add_colorbar: bool = True,
    arrow_scale: float | None = None,
    quiver_kwargs: dict | None = None,
) -> list:
    """Visualize one or more real-valued atomic displacement patterns on a 2D structure.

    In-plane (x, y) displacement -> arrows (quiver).
    Out-of-plane (z) displacement -> marker color (diverging cmap, white = 0).

    This function is purely geometric: it plots a given ``atoms`` structure with a
    given displacement field (e.g. a phonon eigenmode's real part, already sliced
    to whichever atoms/layer the caller wants shown) and has no knowledge of how
    the displacements were obtained. Callers extract per-layer atoms and mode
    displacements (e.g. from an [`Unfold`][unphold.unfold.Unfold] instance's
    Gamma-point eigenvectors) before calling this function.

    Args:
        atoms (aseAtoms): Structure to plot (e.g. one layer of a bilayer system).
        displacements (numpy.ndarray): Displacement field(s) for ``atoms``, shape
            ``(natoms, 3)`` for a single mode or ``(nmodes, natoms, 3)`` for several.
        freqs (float | numpy.ndarray | None): Frequency (or one per mode), used only
            for the subplot title(s) if given.
        axes (Axes | list[Axes]): Axes to draw into: a single ``Axes`` for one mode,
            or a list of ``Axes`` matching ``nmodes``. Must be supplied by the caller;
            this function never creates its own figure.
        cmap (str): Diverging colormap for the out-of-plane displacement.
        norm (Normalize | None): Color normalization. If not given, each mode gets its
            own symmetric normalization (0 at white). Pass a shared ``Normalize`` to
            use one color scale across modes/axes.
        add_colorbar (bool): Whether to add a colorbar to each axes.
        arrow_scale (float | None): Quiver ``scale`` argument (smaller = longer arrows).
        quiver_kwargs (dict | None): Extra kwargs forwarded to ``ax.quiver``.

    Returns:
        list: One scatter mappable per mode/axis, in the same order as ``axes``, so
            callers can build a single shared colorbar externally
            (``fig.colorbar(mappables[0], ax=axes, ...)``).

    Raises:
        ValueError: If ``axes`` is not given, or its length does not match the number
            of modes in ``displacements``.
    """
    if axes is None:
        raise ValueError("`axes` must be supplied by the caller (single Axes or list of Axes)")

    displacements = numpy.asarray(displacements)
    if displacements.ndim == 2:
        displacements = displacements[None, ...]  # (1, natoms, 3)
    nmodes = displacements.shape[0]

    axes_list = [axes] if isinstance(axes, Axes) else list(axes)
    if len(axes_list) != nmodes:
        raise ValueError(f"got {len(axes_list)} axes for {nmodes} mode(s)")

    freqs_list = [freqs] * nmodes if (freqs is None or numpy.isscalar(freqs)) else list(freqs)

    x, y = atoms.get_positions()[:, 0], atoms.get_positions()[:, 1]

    qkw = dict(color="k", width=0.003, zorder=3)
    if arrow_scale is not None:
        qkw["scale"] = arrow_scale
    if quiver_kwargs:
        qkw.update(quiver_kwargs)

    mappables = []
    for mode_idx, ax in enumerate(axes_list):
        fig = ax.get_figure()
        disp = displacements[mode_idx]
        dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]

        mode_norm = norm
        if mode_norm is None:
            vmax = numpy.abs(dz).max()
            mode_norm = Normalize(vmin=-vmax, vmax=vmax)  # symmetric around 0 -> white at 0 for bwr

        sca = ax.scatter(x, y, c=dz, cmap=cmap, norm=mode_norm, s=25, edgecolors="k", linewidths=0.2, zorder=2)
        if add_colorbar:
            fig.colorbar(sca, ax=ax, label="out-of-plane amplitude (z)")

        ax.quiver(x, y, dx, dy, **qkw)

        ax.set_aspect("equal")
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        freq = freqs_list[mode_idx]
        if freq is not None:
            ax.set_title(f"freq = {freq:.4f} THz")

        mappables.append(sca)

    return mappables
